import logging
from abc import abstractmethod

from typing import (TYPE_CHECKING, Dict, List, Optional, Tuple, Union)

import numpy as np
import spacy

if TYPE_CHECKING:
    import spacy  # noqa: F811

logger = logging.getLogger(__name__)


class Neighbors:
    def __init__(self, nlp_obj: 'spacy.language.Language', n_similar: int = 500, w_prob: float = -15.) -> None:
        """
        Initialize class identifying neighbouring words from the embedding for a given word.

        Parameters
        ----------
        nlp_obj
            `spaCy` model.
        n_similar
            Number of similar words to return.
        w_prob
            Smoothed log probability estimate of token's type.
        """
        self.nlp = nlp_obj
        self.w_prob = w_prob
        # list with spaCy lexemes in vocabulary
        # first if statement is a workaround due to some missing keys in models:
        # https://github.com/SeldonIO/alibi/issues/275#issuecomment-665017691
        self.to_check = [self.nlp.vocab[w] for w in self.nlp.vocab.vectors
                         if int(w) in self.nlp.vocab.strings and  # type: ignore[operator]
                         self.nlp.vocab[w].prob >= self.w_prob]
        self.n_similar = n_similar

    def neighbors(self, word: str, tag: str, top_n: int) -> dict:
        """
        Find similar words for a certain word in the vocabulary.

        Parameters
        ----------
        word
            Word for which we need to find similar words.
        tag
            Part of speech tag for the words.
        top_n
            Return only `top_n` neighbors.

        Returns
        -------
        A dict with two fields. The ``'words'`` field contains a `numpy` array of the `top_n` most similar words, \
        whereas the fields ``'similarities'`` is a `numpy` array with corresponding word similarities.
        """

        # the word itself is excluded so we add one to return the expected number of words
        top_n += 1

        texts: List = []
        similarities: List = []
        if word in self.nlp.vocab:
            word_vocab = self.nlp.vocab[word]
            queries = [w for w in self.to_check if w.is_lower == word_vocab.is_lower]
            if word_vocab.prob < self.w_prob:
                queries += [word_vocab]
            by_similarity = sorted(queries, key=lambda w: word_vocab.similarity(w), reverse=True)[:self.n_similar]

            # Find similar words with the same part of speech
            for lexeme in by_similarity:
                # because we don't add the word itself anymore
                if len(texts) == top_n - 1:
                    break
                token = self.nlp(lexeme.orth_)[0]
                if token.tag_ != tag or token.text == word:
                    continue
                texts.append(token.text)
                similarities.append(word_vocab.similarity(lexeme))

        words = np.array(texts) if texts else np.array(texts, dtype='<U')
        return {'words': words, 'similarities': np.array(similarities)}


def load_spacy_lexeme_prob(nlp: 'spacy.language.Language') -> 'spacy.language.Language':
    """
    This utility function loads the `lexeme_prob` table for a spacy model if it is not present.
    This is required to enable support for different spacy versions.
    """
    import spacy
    SPACY_VERSION = spacy.__version__.split('.')
    MAJOR, MINOR = int(SPACY_VERSION[0]), int(SPACY_VERSION[1])

    if MAJOR == 2:
        if MINOR < 3:
            return nlp
        elif MINOR == 3:
            # spacy 2.3.0 moved lexeme_prob into a different package `spacy_lookups_data`
            # https://github.com/explosion/spaCy/issues/5638
            try:
                table = nlp.vocab.lookups_extra.get_table('lexeme_prob')  # type: ignore[attr-defined]
                # remove the default empty table
                if table == dict():
                    nlp.vocab.lookups_extra.remove_table('lexeme_prob')  # type: ignore[attr-defined]
            except KeyError:
                pass
            finally:
                # access the `prob` of any word to load the full table
                assert nlp.vocab["a"].prob != -20.0, f"Failed to load the `lexeme_prob` table for model {nlp}"
    elif MAJOR >= 3:
        # in spacy 3.x we need to manually add the tables
        # https://github.com/explosion/spaCy/discussions/6388#discussioncomment-331096
        if 'lexeme_prob' not in nlp.vocab.lookups.tables:
            from spacy.lookups import load_lookups
            lookups = load_lookups(nlp.lang, ['lexeme_prob'])  # type: ignore[arg-type]
            nlp.vocab.lookups.add_table('lexeme_prob', lookups.get_table('lexeme_prob'))

    return nlp


class AnchorTextSampler:
    @abstractmethod
    def set_text(self, text: str) -> None:
        pass

    @abstractmethod
    def __call__(self, anchor: tuple, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def _joiner(self, arr: np.ndarray, dtype: Optional[str] = None) -> np.ndarray:
        """
        Function to concatenate a `numpy` array of strings along a specified axis.

        Parameters
        ----------
        arr
            1D `numpy` array of strings.
        dtype
           Array type, used to avoid truncation of strings when concatenating along axis.

        Returns
        -------
        Array with one element, the concatenation of the strings in the input array.
        """
        if not dtype:
            return np.array(' '.join(arr))

        return np.array(' '.join(arr)).astype(dtype)


class UnknownSampler(AnchorTextSampler):
    UNK: str = "UNK"  #: Unknown token to be used.

    def __init__(self, nlp: 'spacy.language.Language', perturb_opts: Dict):
        """
        Initialize unknown sampler. This sampler replaces word with the `UNK` token.

        Parameters
        ----------
        nlp
            `spaCy` object.
        perturb_opts
            Perturbation options.
        """
        super().__init__()

        # set nlp and perturbation options
        self.nlp = load_spacy_lexeme_prob(nlp)
        self.perturb_opts: Union[Dict, None] = perturb_opts

        # define buffer for word, punctuation and position
        self.words: List = []
        self.punctuation: List = []
        self.positions: List = []

    def set_text(self, text: str) -> None:
        """
        Sets the text to be processed.

        Parameters
        ----------
        text
            Text to be processed.
        """
        # process text
        processed = self.nlp(text)  # spaCy tokens for text
        self.words = [x.text for x in processed]  # list with words in text
        self.positions = [x.idx for x in processed]  # positions of words in text
        self.punctuation = [x for x in processed if x.is_punct]  # list with punctuation in text

        # set dtype
        self.set_data_type()

    def __call__(self, anchor: tuple, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        The function returns a `numpy` array of `num_samples` where randomly chosen features,
        except those in anchor, are replaced by ``'UNK'`` token.

        Parameters
        ----------
        anchor:
            Indices represent the positions of the words to be kept unchanged.
        num_samples:
            Number of perturbed sentences to be returned.

        Returns
        -------
        raw
            Array containing num_samples elements. Each element is a perturbed sentence.
        data
            A `(num_samples, m)`-dimensional boolean array, where `m` is the number of tokens
            in the instance to be explained.
        """
        assert self.perturb_opts, "Perturbation options are not set."

        # allocate memory for the binary mask and the perturbed instances
        data = np.ones((num_samples, len(self.words)))
        raw = np.zeros((num_samples, len(self.words)), self.dtype)

        # fill each row of the raw data matrix with the text instance to be explained
        raw[:] = self.words

        for i, t in enumerate(self.words):
            # Skip punctuation - don't perturb it (standard NLP practice)
            import string
            if all(c in string.punctuation for c in t):
                continue
            
            # do not perturb words that are in anchor
            # if i in anchor:
                # continue

            # sample the words in the text outside of the anchor that are replaced with UNKs
            n_changed = np.random.binomial(num_samples, self.perturb_opts['sample_proba'])
            changed = np.random.choice(num_samples, n_changed, replace=False)
            raw[changed, i] = UnknownSampler.UNK
            data[changed, i] = 0

        # join the words
        raw = np.apply_along_axis(self._joiner, axis=1, arr=raw, dtype=self.dtype)
        return raw, data

    def set_data_type(self) -> None:
        """
        Working with `numpy` arrays of strings requires setting the data type to avoid
        truncating examples. This function estimates the longest sentence expected
        during the sampling process, which is used to set the number of characters
        for the samples and examples arrays. This depends on the perturbation method
        used for sampling.
        """
        max_len = max(len(self.UNK), len(max(self.words, key=len)))
        max_sent_len = len(self.words) * max_len + len(self.UNK) * len(self.punctuation) + 1
        self.dtype = '<U' + str(max_sent_len)


class SimilaritySampler(AnchorTextSampler):

    def __init__(self, nlp: 'spacy.language.Language', perturb_opts: Dict):
        """
        Initialize similarity sampler. This sampler replaces words with similar words.

        Parameters
        ----------
        nlp
            `spaCy` object.
        perturb_opts
            Perturbation options.

        """
        super().__init__()

        # set nlp and perturbation options
        self.nlp = load_spacy_lexeme_prob(nlp)
        self.perturb_opts = perturb_opts

        # define synonym generator
        self._synonyms_generator = Neighbors(self.nlp)

        # dict containing an np.array of similar words with same part of speech and an np.array of similarities
        self.synonyms: Dict[str, Dict[str, np.ndarray]] = {}
        self.tokens: 'spacy.tokens.Doc'
        self.words: List[str] = []
        self.positions: List[int] = []
        self.punctuation: List['spacy.tokens.Token'] = []

    def set_text(self, text: str) -> None:
        """
        Sets the text to be processed

        Parameters
        ----------
        text
            Text to be processed.
        """
        processed = self.nlp(text)  # spaCy tokens for text
        self.words = [x.text for x in processed]  # list with words in text
        self.positions = [x.idx for x in processed]  # positions of words in text
        self.punctuation = [x for x in processed if x.is_punct]  # punctuation in text
        self.tokens = processed

        # find similar words
        self.find_similar_words()

        # set dtype
        self.set_data_type()

    def find_similar_words(self) -> None:
        """
        This function queries a `spaCy` nlp model to find `n` similar words with the same
        part of speech for each word in the instance to be explained. For each word
        the search procedure returns a dictionary containing a `numpy` array of words (``'words'``)
        and a `numpy` array of word similarities (``'similarities'``).
        """
        for word, token in zip(self.words, self.tokens):
            if word not in self.synonyms:
                self.synonyms[word] = self._synonyms_generator.neighbors(word, token.tag_, self.perturb_opts['top_n'])

    def __call__(self, anchor: tuple, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        The function returns a `numpy` array of `num_samples` where randomly chosen features,
        except those in anchor, are replaced by similar words with the same part of speech of tag.
        See :py:meth:`alibi.explainers.anchors.text_samplers.SimilaritySampler.perturb_sentence_similarity` for details
        of how the replacement works.

        Parameters
        ----------
        anchor:
            Indices represent the positions of the words to be kept unchanged.
        num_samples:
            Number of perturbed sentences to be returned.

        Returns
        -------
        See :py:meth:`alibi.explainers.anchors.text_samplers.SimilaritySampler.perturb_sentence_similarity`.
        """
        assert self.perturb_opts, "Perturbation options are not set."
        return self.perturb_sentence_similarity(anchor, num_samples, **self.perturb_opts)

    def perturb_sentence_similarity(self,
                                    present: tuple,
                                    n: int,
                                    sample_proba: float = 0.5,
                                    forbidden: frozenset = frozenset(),
                                    forbidden_tags: frozenset = frozenset(['PRP$']),
                                    forbidden_words: frozenset = frozenset(['be']),
                                    temperature: float = 1.,
                                    pos: frozenset = frozenset(['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET']),
                                    use_proba: bool = False,
                                    **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perturb the text instance to be explained.

        Parameters
        ----------
        present
            Word index in the text for the words in the proposed anchor.
        n
            Number of samples used when sampling from the corpus.
        sample_proba
            Sample probability for a word if `use_proba=False`.
        forbidden
            Forbidden lemmas.
        forbidden_tags
            Forbidden POS tags.
        forbidden_words
            Forbidden words.
        pos
            POS that can be changed during perturbation.
        use_proba
            Bool whether to sample according to a similarity score with the corpus embeddings.
        temperature
            Sample weight hyper-parameter if ``use_proba=True``.
        **kwargs
            Other arguments. Not used.

        Returns
        -------
        raw
            Array of perturbed text instances.
        data
            Matrix with 1s and 0s indicating whether a word in the text has not been perturbed for each sample.
        """
        # allocate memory for the binary mask and the perturbed instances
        raw = np.zeros((n, len(self.tokens)), self.dtype)
        data = np.ones((n, len(self.tokens)))

        # fill each row of the raw data matrix with the text to be explained
        raw[:] = [x.text for x in self.tokens]

        for i, t in enumerate(self.tokens):  # apply sampling to each token
            # if the word is part of the anchor, move on to next token
            if i in present:
                continue

            # check that token does not fall in any forbidden category
            if (t.text not in forbidden_words and t.pos_ in pos and
                    t.lemma_ not in forbidden and t.tag_ not in forbidden_tags):

                t_neighbors = self.synonyms[t.text]['words']
                # no neighbours with the same tag or word not in spaCy vocabulary
                if t_neighbors.size == 0:
                    continue

                n_changed = np.random.binomial(n, sample_proba)
                changed = np.random.choice(n, n_changed, replace=False)

                if use_proba:  # use similarity scores to sample changed tokens
                    weights = self.synonyms[t.text]['similarities']
                    weights = np.exp(weights / temperature)  # weighting by temperature (check previous implementation)
                    weights = weights / sum(weights)
                else:
                    weights = np.ones((t_neighbors.shape[0],))
                    weights /= t_neighbors.shape[0]

                raw[changed, i] = np.random.choice(t_neighbors, n_changed, p=weights, replace=True)
                data[changed, i] = 0

        raw = np.apply_along_axis(self._joiner, axis=1, arr=raw, dtype=self.dtype)
        return raw, data

    def set_data_type(self) -> None:
        """
        Working with `numpy` arrays of strings requires setting the data type to avoid
        truncating examples. This function estimates the longest sentence expected
        during the sampling process, which is used to set the number of characters
        for the samples and examples arrays. This depends on the perturbation method
        used for sampling.
        """
        max_len = 0
        max_sent_len = 0

        for word in self.words:
            similar_words = self.synonyms[word]['words']
            max_len = max(max_len, int(similar_words.dtype.itemsize /
                                       np.dtype(similar_words.dtype.char + '1').itemsize))
            max_sent_len += max_len
            self.dtype = '<U' + str(max_sent_len)

import shap
class ShapTextSampler(AnchorTextSampler):
    def __init__(self, nlp: 'spacy.language.Language', perturb_opts: Dict):
        super().__init__()
        self.nlp = load_spacy_lexeme_prob(nlp)
        self.model = perturb_opts.get("clf")
        self.perturb_opts = perturb_opts or {}

        self.words: List[str] = []
        self.positions: List[int] = []
        self.punctuation: List = []

        # masker & explainer
        self.masker = None
        self.explainer = None
        self.text = None
        self.predictor = None
        self.shap_values: Union[np.ndarray, None] = None

    def set_text(self, text: str, predictor=None) -> None:
        """Sets the text and prepares SHAP explainer."""
        self.text = text

        # 準備 explainer
        if predictor is not None:
            self.predictor = predictor

        if self.explainer is None and self.predictor is not None:
            self.masker = shap.maskers.Text(tokenizer=self.SpacyTokenizerWrapper(self.nlp))
            self.explainer = shap.Explainer(
                self.predict_fn, 
                self.masker, 
                algorithm="permutation"
            )

        # tokenization
        if self.nlp is not None:  # 有 spaCy，用 self.nlp 處理
            processed = self.nlp(text)
            self.words = [x.text for x in processed]
            self.positions = [x.idx for x in processed]
            self.punctuation = [x for x in processed if x.is_punct]
        else:  # 沒有 spaCy，用 split()
            self.words = text.split()
            self.positions = list(range(len(self.words)))
            self.punctuation = []

        # dtype 設定
        self.set_data_type()

        # 計算 shap 值
        shap_out = self.explainer([self.text])
        shap_values = shap_out.values
        if shap_values.ndim == 3:  
            shap_values = shap_values[0, :, 1] # 多分類
        else:
            shap_values = shap_values[0, :] # 二分類
        self.shap_values = shap_values


    def __call__(self, anchor: tuple, num_samples: int):
        """根據 SHAP 擾動樣本"""
        assert self.text is not None, "請先呼叫 set_text()"

        n_tokens = len(self.words)
        raw = np.zeros((num_samples, n_tokens), self.dtype)
        data = np.ones((num_samples, n_tokens))
        raw[:] = self.words

        # SHAP 值轉換為機率 (越大越高機率被遮蔽)
        shap_abs = np.abs(self.shap_values)
        perturb_proba = shap_abs / (shap_abs.sum() + 1e-6)

        # 排序 shap 值
        sorted_idx = np.argsort(-np.abs(self.shap_values))

        for j in range(num_samples):
            max_mask = max(1, int(0.2 * n_tokens))
            n_mask = np.random.randint(1, max_mask + 1)

            # 排除 anchor
            candidate = [i for i in sorted_idx if i not in anchor]
            if not candidate:
                continue

            # 根據 shap 機率抽樣
            p = perturb_proba[candidate]
            p /= p.sum()  # normalize
            n_mask = min(n_mask, len(candidate))

            # 有權重的隨機抽樣 → 重要 token 較容易被挑中
            masked_indices = np.random.choice(candidate, size=n_mask, replace=False, p=p)

            # 遮掉選中的 token
            for i in masked_indices:
                raw[j, i] = "UNK"
                data[j, i] = 0

        raw = np.apply_along_axis(self._joiner, axis=1, arr=raw, dtype=self.dtype)
        return raw, data
    
    def set_data_type(self) -> None:
        """
        設定 numpy 字串 dtype，避免截斷。
        """
        max_len = max(len("UNK"), len(max(self.words, key=len)))
        max_sent_len = len(self.words) * max_len + len("UNK") * len(self.punctuation) + 1
        self.dtype = '<U' + str(max_sent_len)

    def predict_fn(self, texts):
        """SHAP explainer 用的預測函式，確保輸入為向量化後的 2D array"""
        vectorizer = self.perturb_opts.get("vectorizer", None)
        if vectorizer is not None:
            X = vectorizer.transform(texts)
        else:
            # 如果沒有傳入 vectorizer，就假設模型能處理原始文字
            X = texts
        return self.model.predict_proba(X)[:, 1]

    
    class SpacyTokenizerWrapper:
        def __init__(self, nlp):
            self.nlp = nlp
            self.current_tokens = []  # 初始化，避免 AttributeError

        def __call__(self, text):
            """SHAP 會呼叫這個來做 tokenization"""
            doc = self.nlp(text)
            tokens = [t.text for t in doc]
            self.current_tokens = tokens  # 保存給 decode 用
            return {"input_ids": list(range(len(tokens))), "tokens": tokens}

        def decode(self, ids):
            """把 token id 轉回文字（SHAP 用來還原句子）"""
            if not hasattr(self, "current_tokens") or not self.current_tokens:
                # fallback 機制，避免一開始 decode 前沒有 token
                return " ".join([f"[{i}]" for i in ids])
            tokens = [self.current_tokens[i] for i in ids if i < len(self.current_tokens)]
            return " ".join(tokens)



import shap
import numpy as np

class ShapTabularSampler:
    def __init__(self, model, train_data, feature_names, categorical_features=None):
        self.model = model
        self.train_data = train_data
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.explainer = shap.Explainer(model.predict_proba, train_data)
        self.shap_values = None
        self.instance_label = None

    def set_instance_label(self, X):
        self.instance_label = self.model.predict(X.reshape(1, -1))[0]
        shap_out = self.explainer(X.reshape(1, -1))
        shap_vals = shap_out.values
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[0, :, 1]
        else:
            shap_vals = shap_vals[0, :]
        self.shap_values = np.abs(shap_vals)
        self.importance = (self.shap_values.max() - self.shap_values)
        self.importance /= (self.importance.max() + 1e-6)

    def __call__(self, anchor, num_samples):
        assert self.shap_values is not None, "請先呼叫 set_instance_label()"

        n_feats = len(self.feature_names)
        data = np.ones((num_samples, n_feats))
        raw = np.zeros((num_samples, n_feats))
        raw[:] = self.train_data[np.random.choice(len(self.train_data), num_samples)]

        for j in range(num_samples):
            for i in range(n_feats):
                if i in anchor:
                    continue
                if np.random.rand() < self.importance[i]:
                    # 根據 SHAP 值擾動數值特徵
                    if i not in self.categorical_features:
                        mean, std = self.train_data[:, i].mean(), self.train_data[:, i].std()
                        raw[j, i] = np.clip(np.random.normal(mean, std), self.train_data[:, i].min(), self.train_data[:, i].max())
                    else:
                        # 類別特徵改為隨機取不同類別
                        raw[j, i] = np.random.choice(np.unique(self.train_data[:, i]))
                    data[j, i] = 0
        return raw, data


import random
import numpy as np
from copy import deepcopy

class EditDistanceTextSampler:
    """
    Edit-distance based perturbation sampler for AnchorText.
    Similar to TabularSampler logic, but applied to text tokens.

    Operations: substitute / insert / delete
    Anchor tokens will NOT be modified.
    """

    def __init__(self, words, seed=0, edit_dist=1, vocab=None):
        """
        Args:
            words: tokenized original sentence (list[str])
            edit_dist: radius of edit operations
            vocab: optional vocabulary for insertion/substitution; 
                   defaults to unique tokens in sentence
        """
        random.seed(seed)
        np.random.seed(seed)

        self.words = words
        self.edit_dist = edit_dist
        self.vocab = vocab if vocab else list(set(words))

    def set_text(self, text):
        """Update the text to be perturbed."""
        self.words = text.split()
        # Update vocab if it was empty and now we have words
        if not self.vocab and self.words:
            self.vocab = list(set(self.words))

    def perturb_once(self, anchor_positions=None):
        """
        Create ONE perturbed text based on edit distance operations.
        
        Parameters:
            anchor_positions: set of positions NOT to modify. If None, any position can be modified.
        """
        if anchor_positions is None:
            anchor_positions = set()  # No protected positions
        
        # Safety check: if vocab is empty, don't perturb
        if not self.vocab or not self.words:
            return self.words.copy(), [0] * len(self.words)
            
        new_words = deepcopy(self.words)
        
        # mask: track which positions were modified (always matches new_words length)
        mask = [0] * len(new_words)

        for _ in range(self.edit_dist):
            # Safety check in loop
            if not new_words:
                break
                
            # Select valid operations based on vocab availability and word count
            valid_ops = ["substitute"]
            if self.vocab:
                valid_ops.append("insert")
            # Allow delete only if we have >1 words to avoid reducing to empty
            if len(new_words) > 1:
                valid_ops.append("delete")
            
            # If no valid ops available, skip
            if not valid_ops:
                break
            
            op = random.choice(valid_ops)

            # select position not in anchor (or any position if no anchor)
            if anchor_positions:
                valid_positions = [i for i in range(len(new_words)) if i not in anchor_positions]
            else:
                valid_positions = list(range(len(new_words)))

            if len(valid_positions) == 0:
                break

            pos = random.choice(valid_positions)

            # ==== EDIT OPS ====

            # Substitute
            # if op == "substitute":
            new_word = random.choice(self.vocab)
            new_words[pos] = new_word
            mask[pos] = 1

            # # Insert
            # elif op == "insert":
            #     new_word = random.choice(self.vocab)
            #     new_words.insert(pos, new_word)
            #     mask.insert(pos, 1)

            # # Delete
            # elif op == "delete":
            #     del new_words[pos]
            #     del mask[pos]

        return new_words, mask

    def __call__(self, anchor=None, num_samples=100):
        """
        Generate (raw_texts, masks).
        
        Parameters:
            anchor: tuple (indices, positions) for anchor-based perturbation. 
                   If None, random perturbation without anchor constraints.
            num_samples: number of samples to generate
        """
        if anchor is None:
            # No anchor - pure random perturbation
            anchor_positions = None
        else:
            # Anchor-based perturbation
            _, anchor_tokens = anchor
            anchor_positions = set(anchor_tokens)

        raw_texts = []
        masks = []

        for _ in range(num_samples):
            pert, mask = self.perturb_once(anchor_positions)
            raw_texts.append(" ".join(pert))
            masks.append(mask)

        return np.array(raw_texts, dtype=object), np.array(masks, dtype=object)