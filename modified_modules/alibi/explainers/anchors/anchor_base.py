# import copy
# import itertools
import logging
from collections import defaultdict, namedtuple
# import random
import random
from typing import Callable, Dict, List, Optional, Set, Tuple
import numpy as np
from datasets.og_loader import OGD2
from learner.dfa_learner import make_dfa_complete, trim_dfa
from automaton.utils import plot_dfa_beam_stats
from modified_modules.alibi.utils.distributions import kl_bernoulli
from learner import AUTO_INSTANCE

logger = logging.getLogger(__name__)

# TODO: Discuss logging strategy

class AnchorBaseBeam:

    def __init__(self, samplers: List[Callable], dfas=None, predictor=None, **kwargs) -> None:
        """
        Parameters
        ---------
        samplers
            Objects that can be called with args (`result`, `n_samples`) tuple to draw samples.
        """
        self.samplers = samplers
        if isinstance(samplers, list) and len(samplers) > 1:
            self.sample_fcn = samplers
        else:
            self.sample_fcn = samplers[0] if isinstance(samplers, list) else samplers

        # Initial size (in batches) of data/raw data samples cache.
        self.sample_cache_size = kwargs.get('sample_cache_size', 1000)

        # when only the max of self.margin or batch size remain emptpy, the cache is
        # extended to accommodate an additional sample_cache_size batches.
        self.margin = kwargs.get('cache_margin', 100)

        self.type = ''
        self.testing_data = []
        self.dfas = dfas or []
        self.predictor = predictor
        self.iteration = 0

    def _init_state(self, batch_size: int) -> None:
        """
        Initialises the object state, which is used to compute result accuracies & accuracy bounds
        and provide metadata for explanation objects.

        Parameters
        ----------
        batch_size
            See :py:meth:`alibi.explainers.anchors.anchor_base.AnchorBaseBeam.anchor_beam` method.
        coverage_data
            See :py:meth:`alibi.explainers.anchors.anchor_base.AnchorBaseBeam._get_coverage_samples` method.
        """

        prealloc_size = batch_size * self.sample_cache_size
        # Initialize data as a list to support variable-length sequences
        data_init = []
        
        # t_ indicates that the attribute is a dictionary with entries for each anchor
        self.state: dict = {
            't_coverage': defaultdict(lambda: 0.),  # anchors' coverage
            't_coverage_idx': defaultdict(set),  # index of anchors in coverage set
            't_covered_true': defaultdict(None),  # samples with same pred as instance where t_ applies
            't_covered_false': defaultdict(None),  # samples with dif pred to instance where t_ applies
            't_idx': defaultdict(set),  # row idx in sample cache where the anchors apply
            't_nsamples': defaultdict(lambda: 0.),  # total number of samples drawn for the anchors
            't_accepted': defaultdict(lambda: 0.),  # total number of samples drawn for the accepted traces of dfa
            't_order': defaultdict(list),  # anchors are sorted to avoid exploring permutations
            # this is the order in which anchors were found
            't_positives': defaultdict(lambda: 0.),  # nb of samples where result pred = pred on instance
            't_negatives': defaultdict(lambda: 0.),
            'prealloc_size': prealloc_size,  # samples caches size
            'data': data_init,  # samples caches (list for variable-length support)
            'labels': np.zeros(prealloc_size, ),  # clf pred labels on raw_data
            'current_idx': 0,
            # 'coverage_data': coverage_data,  # coverage data
            # 'coverage_raw': raw_cov_data,  # raw coverage data
            # 'coverage_label': coverage_label,  # coverage label
        }
        self.state['t_order'][()] = ()  # Trivial order for the empty result

    @staticmethod
    def _sort(x: tuple, allow_duplicates=False) -> tuple:
        """
        Sorts a tuple, optionally removing duplicates.

        Parameters
        ----------
        x:
            Tuple to be sorted.
        allow_duplicates:
            If ``True``, duplicate entries are kept.

        Returns
        -------
        A sorted tuple.
        """

        if allow_duplicates:
            return tuple(sorted(x))

        return tuple(sorted(set(x)))

    @staticmethod
    def dup_bernoulli(p: np.ndarray, level: np.ndarray, n_iter: int = 17) -> np.ndarray:
        """
        Update upper accuracy bound for a candidate anchors dependent on the KL-divergence.

        Parameters
        ----------
        p
            Accuracy of candidate anchors.
        level
            `beta / nb of samples` for each result.
        n_iter
            Number of iterations during lower bound update.

        Returns
        -------
        Updated upper accuracy bounds array.
        """
        # TODO: where does 17x sampling come from?
        lm = p.copy()
        um = np.minimum(np.minimum(p + np.sqrt(level / 2.), 1.0), 1.0)

        # Perform bisection algorithm to find the largest qm s.t. kl divergence is > level
        for j in range(1, n_iter):
            qm = (um + lm) / 2.
            kl_gt_idx = kl_bernoulli(p, qm) > level
            kl_lt_idx = np.logical_not(kl_gt_idx)
            um[kl_gt_idx] = qm[kl_gt_idx]
            lm[kl_lt_idx] = qm[kl_lt_idx]

        return um

    @staticmethod
    def dlow_bernoulli(p: np.ndarray, level: np.ndarray, n_iter: int = 17) -> np.ndarray:
        """
        Update lower accuracy bound for a candidate anchors dependent on the KL-divergence.

        Parameters
        ----------
        p
            Accuracy of candidate anchors.
        level
            `beta / nb of samples` for each result.
        n_iter
            Number of iterations during lower bound update.

        Returns
        -------
        Updated lower accuracy bounds array.
        """

        um = p.copy()
        lm = np.clip(p - np.sqrt(level / 2.), 0.0, 1.0)  # lower bound

        # Perform bisection algorithm to find the smallest qm s.t. kl divergence is > level
        for _ in range(1, n_iter):
            qm = (um + lm) / 2.
            kl_gt_idx = kl_bernoulli(p, qm) > level
            kl_lt_idx = np.logical_not(kl_gt_idx)
            lm[kl_gt_idx] = qm[kl_gt_idx]
            um[kl_lt_idx] = qm[kl_lt_idx]

        return lm

    @staticmethod
    def compute_beta(n_features: int, t: int, delta: float) -> float:
        """
        Parameters
        ----------
        n_features
            Number of candidate anchors.
        t
            Iteration number.
        delta
            Confidence budget, candidate anchors have close to optimal precisions with prob. `1 - delta`.

        Returns
        -------
        Level used to update upper and lower accuracy bounds.
        """
        # The following constants are defined and used in the paper introducing the KL-LUCB bandit algorithm
        # (http://proceedings.mlr.press/v30/Kaufmann13.html). Specifically, Theorem 1 proves the lower bounds
        # of these constants to ensure the algorithm is PAC with probability at least 1-delta. Also refer to
        # section "5. Numerical experiments" where these values are used empirically.
        alpha = 1.1
        k = 405.5
        temp = np.log(k * n_features * (t ** alpha) / delta)

        return temp + np.log(temp)

    def _get_coverage_samples(self, coverage_samples: int, samplers: Optional[List[Callable]] = None) -> np.ndarray:
        """
        Draws samples uniformly at random from the training set.

        Parameters
        ---------
        coverage_samples
            See :py:meth:`alibi.explainers.anchors.anchor_base.AnchorBaseBeam.anchor_beam` method.
        samplers
            See :py:meth:`alibi.explainers.anchors.anchor_base.AnchorBaseBeam.__init__` method.

        Returns
        -------
        coverage_data
            Binarised samples, where 1 indicates the feature has same value/is in same beam as
            instance to be explained. Used to determine, e.g., which samples an result applies to.
        """
        if self.type == 'Text':
            raw_cov, preds, coverage_data = samplers((0, ()), coverage_samples, compute_labels=True)
        else:
            raw_cov, _, _, preds, coverage_data, _, _  = samplers((0, ()), coverage_samples, compute_labels=True)
        return raw_cov, preds, coverage_data

    def select_critical_arms(self, means: np.ndarray, ub: np.ndarray, lb: np.ndarray, n_samples: np.ndarray,
                             delta: float, top_n: int, t: int):
        """
        Determines a set of two anchors by updating the upper bound for low empirical accuracy anchors and
        the lower bound for anchors with high empirical accuracy.

        Parameters
        ----------
        means
            Empirical mean result accuracies.
        ub
            Upper bound on result accuracies.
        lb
            Lower bound on result accuracies.
        n_samples
            The number of samples drawn for each candidate result.
        delta
            Confidence budget, candidate anchors have close to optimal accuracies with prob. `1 - delta`.
        top_n
            Number of arms to be selected.
        t
            Iteration number.

        Returns
        -------
        Upper and lower accuracy bound indices.
        """

        crit_arms = namedtuple('crit_arms', ['ut', 'lt'])

        sorted_means = np.argsort(means)  # ascending sort of result candidates by accuracy
        beta = self.compute_beta(len(means), t, delta)

        # J = the beam width top result candidates with highest accuracy
        # not_J = the rest
        J = sorted_means[-top_n:]
        not_J = sorted_means[:-top_n]

        # update upper bound for lowest accuracy result candidates
        ub[not_J] = self.dup_bernoulli(means[not_J], beta / n_samples[not_J])
        # update lower bound for highest accuracy result candidates
        lb[J] = self.dlow_bernoulli(means[J], beta / n_samples[J])

        # for the low accuracy result candidates, compute the upper accuracy bound and keep the index ...
        # ... of the result candidate with the highest upper accuracy value -> ut
        # for the high accuracy result candidates, compute the lower accuracy bound and keep the index ...
        # ... of the result candidate with the lowest lower accuracy value -> lt
        ut = not_J[np.argmax(ub[not_J])]
        lt = J[np.argmin(lb[J])]

        return crit_arms._make((ut, lt))

    
    def kllucb_automata(self, dfas: list, init_stats: dict, epsilon: float, delta: float, batch_size: int, top_n: int,
               verbose: bool = False, verbose_every: int = 1) -> np.ndarray:
        """
        Implements the KL-LUCB algorithm (Kaufmann and Kalyanakrishnan, 2013).

        Parameters
        ----------
        dfas:
            A list of dfas from which two critical dfas are selected (see Kaufmann and Kalyanakrishnan, 2013).
        init_stats
            Dictionary with lists containing nb of samples used and where sample predictions equal the desired label.
        epsilon
            Accuracy bound tolerance for convergence.
        delta
            Used to compute `beta`.
        batch_size
            Number of samples.
        top_n
            Min of beam width size or number of candidate dfas.
        verbose
            Whether to print intermediate output.
        verbose_every
            Whether to print intermediate output every `verbose_every` steps.

        Returns
        -------
        Indices of best result options. Number of indices equals min of beam width or nb of candidate dfas.
        """

        # n_features equals to the nb of candidate dfas
        n_features = len(dfas)

        # arrays for total number of samples & positives (# samples where prediction equals desired label)
        n_samples, n_accepted, positives, negatives = init_stats['n_samples'], init_stats['n_accepted'], init_stats['positives'], init_stats['negatives']
        dfas_to_sample, dfas_idx = [], []
        for f in np.where(n_samples == 0)[0]:
            dfas_to_sample.append(dfas[f])
            dfas_idx.append(f)

        if dfas_idx:
            true_accept, false_reject, total, accepted = self.draw_samples(dfas_to_sample, 1)
            positives[dfas_idx] += true_accept
            negatives[dfas_idx] += false_reject
            n_samples[dfas_idx] += total
            n_accepted[dfas_idx] += accepted

        if n_features == top_n:  # return all options b/c of beam search width
            return np.arange(n_features)

        # update the upper and lower accuracy bounds until the difference between the best upper ...
        # ... accuracy bound of the low accuracy dfas and the worst lower accuracy bound of the high ...
        # ... accuracy dfas is smaller than eps
        denom = positives + negatives
        means = np.divide(
            denom,
            n_samples,
            out=np.zeros_like(positives),
            where=n_samples != 0
        )
        ub, lb = np.zeros(n_samples.shape), np.zeros(n_samples.shape)
        t = 1
        crit_a_idx = self.select_critical_arms(means, ub, lb, n_samples, delta, top_n, t)
        B = ub[crit_a_idx.ut] - lb[crit_a_idx.lt]
        verbose_count = 0

        MAX_ROUNDS = 500 
        while B > epsilon and t < MAX_ROUNDS:
            print(f"Round {t} : {crit_a_idx}")
            verbose_count += 1
            if verbose and verbose_count % verbose_every == 0:
                ut, lt = crit_a_idx
                print('Best: %d (mean:%.10f, n: %d, lb:%.4f)' %
                      (lt, means[lt], n_samples[lt], lb[lt]), end=' ')
                print('Worst: %d (mean:%.4f, n: %d, ub:%.4f)' %
                      (ut, means[ut], n_samples[ut], ub[ut]), end=' ')
                print('B = %.2f' % B)

            # draw samples for each critical result, update dfas' mean, upper and lower
            # bound accuracy estimate
            selected_dfas = [dfas[idx] for idx in crit_a_idx]
            true_accept, false_reject, total, accepted = self.draw_samples(selected_dfas, batch_size)
            idx = list(crit_a_idx)
            positives[idx] += true_accept
            negatives[idx] += false_reject
            n_samples[idx] += total
            n_accepted[idx] += accepted
            denom = positives + negatives
            means = np.divide(
                denom,
                n_samples,
                out=np.zeros_like(positives),
                where=n_samples != 0
            )
            t += 1
            crit_a_idx = self.select_critical_arms(means, ub, lb, n_samples, delta, top_n, t)
            B = ub[crit_a_idx.ut] - lb[crit_a_idx.lt]

        sorted_means = np.argsort(means)

        return sorted_means[-top_n:]

    def draw_samples(self, dfas: list, batch_size: int) -> Tuple[tuple, tuple]:
        """
        Parameters
        ----------
        dfas
            DFAs on which samples are conditioned.
        batch_size
            The number of samples drawn for each result.

        Returns
        -------
        A tuple of positive samples (for which prediction matches desired label) and a tuple of \
        total number of samples drawn.
        """
        sample_stats: List = []
        pos: Tuple = tuple()
        total: Tuple = tuple()
        samples_iter = [self.sample_fcn(num_samples=batch_size) for dfa in dfas]
        
        for samples, dfa in zip(samples_iter, dfas):
            raw_data, labels = samples

            # update state records
            sample_stats.append(self.update_state(labels, raw_data, dfa))
            true_accept, false_reject, total, accepted = list(zip(*sample_stats))

        return true_accept, false_reject, total, accepted

    def update_state(self, labels: np.ndarray,
                     samples: Tuple[np.ndarray, float], dfa: tuple) -> Tuple[int, int]:
        """raw_data: np.ndarray, 
        Updates the explainer state (see :py:meth:`alibi.explainers.anchors.anchor_base.AnchorBaseBeam.__init__`
        for full state definition).

        Parameters
        ----------
        covered_true
            Examples where the result applies and the prediction is the same as on
            the instance to be explained.
        covered_false
            Examples where the result applies and the prediction is the different to
            the instance to be explained.
        samples
            A tuple containing discretized data, coverage and the result sampled.
        labels
            An array indicating whether the prediction on the sample matches the label
            of the instance to be explained.
        dfa
            The result to be updated.

        Returns
        -------
        A tuple containing the number of instances equals desired label of observation \
        to be explained the total number of instances sampled, and the result that was sampled.
        """

        # data = binary matrix where 1 means a feature has the same value as the feature in the result
        n_samples = len(samples)
        current_idx = self.state['current_idx']
        accepts = np.array([AUTO_INSTANCE.check_path_accepted(dfa, p) for p in samples])
        true_accept = np.sum((labels == 1) & (accepts == True))
        false_reject = np.sum((labels == 0) & (accepts == False))        
        idxs = range(current_idx, current_idx + n_samples)
        self.state['t_idx'][id(dfa)].update(idxs)
        self.state['t_nsamples'][id(dfa)] += n_samples
        self.state['t_accepted'][id(dfa)] += np.sum(accepts)
        self.state['t_positives'][id(dfa)] += true_accept
        self.state['t_negatives'][id(dfa)] += false_reject
        # Store samples in list (support variable-length sequences)
        self.state['data'].extend(samples)
        # Extend labels array if needed
        if current_idx + n_samples > len(self.state['labels']):
            self.state['labels'] = np.hstack(
                (self.state['labels'], np.zeros(n_samples, labels.dtype))
            )
        self.state['labels'][current_idx:current_idx + n_samples] = labels
        self.state['current_idx'] += n_samples

        return true_accept.sum(), false_reject.sum(), n_samples, np.sum(accepts)

    def get_init_stats(self, dfas: list, coverages=False) -> dict:
        """
        Finds the number of samples already drawn for each result in dfas, their
        comparisons with the instance to be explained and, optionally, coverage.

        Parameters
        ----------
        dfas
            Candidate dfas.
        coverages
            If ``True``, the statistics returned contain the coverage of the specified dfas.

        Returns
        -------
        Dictionary with lists containing nb of samples used and where sample predictions equal the desired label.
        """

        def array_factory(size: tuple):
            return lambda: np.zeros(size)

        state = self.state
        stats: Dict[str, np.ndarray] = defaultdict(array_factory((len(dfas),)))
        for i, dfa in enumerate(dfas):
            stats['n_samples'][i] = state['t_nsamples'][id(dfa)]
            stats['positives'][i] = state['t_positives'][id(dfa)]
            stats['negatives'][i] = state['t_negatives'][id(dfa)]
            stats['n_accepted'][i] = state['t_accepted'][id(dfa)]
            if coverages:
                stats['coverages'][i] = state['t_coverage'][id(dfa)]

        return stats
    
    def get_automata_metadata(self, automata, success, batch_size: int = 100) -> dict:
        """
        取得 DFA 的精確度、覆蓋率與範例資訊
        """
        state = self.state
        automata_id = id(automata)

        automata_metadata: dict = {
            'automata': automata,
            'training_accuracy': [],
            'testing_accuracy': [],
            'coverage': [],
            'size': [],
            'examples': [],
            'num_preds': len(state['data']),
            'success': success,
            'false_accept': [],
            'true_reject': []
        }

        # get training accuracy
        if automata_id not in state['t_accepted'] or state['t_accepted'][automata_id] == 0:
            true_accept, false_reject, total, accepted = self.draw_samples([automata], batch_size)
            state['t_positives'][automata_id] += true_accept
            state['t_negatives'][automata_id] += false_reject
            state['t_nsamples'][automata_id] += total[0]
            state['t_accepted'][automata_id] += accepted

        positives = state['t_positives'][automata_id]
        negatives = state['t_negatives'][automata_id]
        total = state['t_nsamples'][automata_id]
        denom = positives + negatives
        mean_accuracy = (denom / total) if total > 0 else 0.0
        automata_metadata['training_accuracy'].append(mean_accuracy)

        # get testing accuracy (only if we have a predictor)
        if self.predictor is not None and len(self.testing_data) > 0:
            accepts = np.array([AUTO_INSTANCE.check_path_accepted(automata, p) for p in self.testing_data])
            labels = self.predictor(self.testing_data)
            
            # Get instance label - handle both Tabular and Text types
            instance_label = 1  # default
            if hasattr(self.sample_fcn, 'tab_sampler') and hasattr(self.sample_fcn.tab_sampler, 'instance_label'):
                instance_label = self.sample_fcn.tab_sampler.instance_label
            
            true_accept = np.sum((labels == instance_label) & (accepts == True))
            false_reject = np.sum((labels != instance_label) & (accepts == False)) 
            total = len(labels)
            denom = true_accept + false_reject
            mean_accuracy = (denom / total) if total > 0 else 0.0
            automata_metadata['testing_accuracy'].append(mean_accuracy)  

            false_accept = [p for p, lab, acc in zip(self.testing_data, labels, accepts) if (lab != instance_label) and acc]
            true_reject = [p for p, lab, acc in zip(self.testing_data, labels, accepts) if (lab == instance_label) and acc]  
            automata_metadata['false_accept'].append(tuple(false_accept))
            automata_metadata['true_reject'].append(tuple(true_reject))
        else:
            # If no predictor or testing data, set testing accuracy to training accuracy
            automata_metadata['testing_accuracy'].append(mean_accuracy)
            automata_metadata['false_accept'].append(tuple())
            automata_metadata['true_reject'].append(tuple())

        if hasattr(automata, 'size'):
            size = automata.size
        elif hasattr(automata, 'states'):
            size = len(automata.states)
        else:
            size = None  # or set to 0 or raise a warning
        automata_metadata['size'].append(size)
        return automata_metadata

    @staticmethod
    def to_sample(means: np.ndarray, ubs: np.ndarray, lbs: np.ndarray, desired_confidence: float, epsilon_stop: float):
        """
        Given an array of mean result accuracies and their upper and lower bounds, determines for which anchors
        more samples need to be drawn in order to estimate the anchors accuracy with `desired_confidence` and error
        tolerance.

        Parameters
        ----------
        means:
            Mean accuracies (each element represents a different result).
        ubs:
            Accuracies' upper bounds (each element represents a different result).
        lbs:
            Accuracies' lower bounds (each element represents a different result).
        desired_confidence:
            Desired level of confidence for accuracy estimation.
        epsilon_stop:
            Tolerance around desired accuracy.

        Returns
        -------
        Boolean array indicating whether more samples are to be drawn for that particular result.
        """

        return ((means >= desired_confidence) & (lbs < desired_confidence - epsilon_stop)) | \
               ((means < desired_confidence) & (ubs >= desired_confidence + epsilon_stop))

    def anchor_beam(self, type:str, alphabet: List = [], automaton_type: str = "DFA", delta: float = 0.05, epsilon: float = 0.1, accuracy_threshold: float = 1., state_threshold: int = 5,
                    select_by: str = "accuracy",
                    beam_size: int = 1, epsilon_stop: float = 0.05, min_samples_start: int = 100,
                    min_anchor_size: Optional[int] = None, stop_on_first: bool = False, batch_size: int = 100,
                    coverage_samples: int = 10000, verbose: bool = False, verbose_every: int = 1, 
                    preinitialized: bool = False,
                    output_dir: str = "test_result/explain",
                    init_num_samples: int = 1000,
                    **kwargs) -> dict:

        """
        Uses the KL-LUCB algorithm (Kaufmann and Kalyanakrishnan, 2013) together with additional sampling to search
        feature sets (anchors) that guarantee the prediction made by a classifier model. The search is greedy if
        ``beam_size=1``. Otherwise, at each of the `max_anchor_size` steps, `beam_size` solutions are explored.
        By construction, solutions found have high accuracy (defined as the expected of number of times the classifier
        makes the same prediction when queried with the feature subset combined with arbitrary samples drawn from a
        noise distribution). The algorithm maximises the coverage of the solution found - the frequency of occurrence
        of records containing the feature subset in set of samples.

        Parameters
        ----------
        delta
            Used to compute `beta`.
        epsilon
            Accuracy bound tolerance for convergence.
        desired_confidence
            Desired level of accuracy (`tau` in `paper <https://homes.cs.washington.edu/~marcotcr/aaai18.pdf>`_).
        beam_size
            Beam width.
        epsilon_stop
            Confidence bound margin around desired accuracy.
        min_samples_start
            Min number of initial samples.
        max_anchor_size
            Max number of features in result.
        stop_on_first
            Stop on first valid result found.
        coverage_samples
            Number of samples from which to build a coverage set.
        batch_size
            Number of samples used for an arm evaluation.
        verbose
            Whether to print intermediate LUCB & anchor selection output.
        verbose_every
            Print intermediate output every verbose_every steps.

        Returns
        -------
        Explanation dictionary containing anchors with metadata like coverage and accuracy and examples.
        """

        self.type = type
        # ====== Automaton Factory ======
        def automaton_factory(automaton_type: str):
            if automaton_type.upper() == "DFA":
                from learner.dfa_learner import DFALearner
                return DFALearner()
            elif automaton_type.upper() == "RA":
                from learner.ra_learner import RegisterAutomataLearner
                ra_theory = kwargs.get('theory', 'Integer')
                ra_num_states = kwargs.get('num_states', 8)
                ra_num_registers = kwargs.get('num_registers', 2)
                ra_constants = kwargs.get('constants', None)
                if ra_constants is None:
                    ra_constants = []
                print(f"[DEBUG] RA params: theory={ra_theory}, states={ra_num_states}, registers={ra_num_registers}, constants={ra_constants}")
                return RegisterAutomataLearner(
                    constants=ra_constants,
                    theory=ra_theory,
                    num_states=ra_num_states,
                    num_registers=ra_num_registers
                )
            else:
                raise ValueError(f"Unknown automaton type: {automaton_type}")

        global AUTO_INSTANCE
        AUTO_INSTANCE = automaton_factory(automaton_type)

        if hasattr(AUTO_INSTANCE, 'get_sampler'):
            sampler_cls = AUTO_INSTANCE.get_sampler()
            accuracy_sampler = next(s for s in self.samplers if isinstance(s, sampler_cls))
        else:
            accuracy_sampler = self.samplers[0]
        inti_samples, inti_label = accuracy_sampler(num_samples=init_num_samples, compute_labels=True)
       
        self.samplers = [accuracy_sampler]
        self.sample_fcn = accuracy_sampler
        self.iteration = 0
        self._init_state(batch_size)

        # Create initial automaton using positive and negative samples
        positive_samples = [x for x, y in zip(inti_samples, inti_label) if y == 1]
        negative_samples = [x for x, y in zip(inti_samples, inti_label) if y == 0]
        origin_automata = AUTO_INSTANCE.create_init_automata(type, positive_samples, negative_samples)
        self.automatas = [origin_automata]
        self.testing_data = inti_samples

        # sample by default 1 or min_samples_start more random value(s)
        (true_accept,), (false_reject,), (total,), (accepted,) = self.draw_samples([self.automatas[0]], min_samples_start)

        # mean = fraction of labels sampled data that equals the label of the instance to be explained, ...
        # ... equivalent to prec(A) in paper (eq.2)
        mean = np.array([(true_accept + false_reject) / total] if total > 0 else [0.0])
        beta = np.log(1. / delta)
        # lower bound on mean accuracy
        lb = self.dlow_bernoulli(mean, np.array([beta / total]))

        # if lower accuracy bound below tau with margin eps, keep sampling data until lb is high enough ...
        # or mean falls below accuracy threshold
        while mean > accuracy_threshold and lb < accuracy_threshold - epsilon:
            (n_true_accept,), (n_false_reject,), (n_total,), (n_accepted,) = self.draw_samples([self.automatas[0]], batch_size)
            true_accept += n_true_accept
            false_reject += n_false_reject
            total += n_total
            accepted += n_accepted
            mean = np.array([(true_accept + false_reject) / total] if total > 0 else [0.0])
            lb = self.dlow_bernoulli(mean, np.array([beta / total]))
        
        # Robustly get automaton size: use .size if present, else len(states), else fallback
        automaton = self.automatas[0]
        if hasattr(automaton, 'size'):
            size = automaton.size
        elif hasattr(automaton, 'states'):
            size = len(automaton.states)
        else:
            size = None  # or set to 0 or raise a warning

        # if prec_lb(A) > tau for A=() then the empty result satisfies the constraints ...
        min_states = 2
        if lb > accuracy_threshold and size <= state_threshold and size >= min_states:
            return {
                'automata': self.automatas[0],
                'training_accuracy': mean,
                'testing_accuracy': 1.0,
                'size': size,
                'coverage': [],
                'examples': [],
                'success': True,
                'false_accept': [],
                'true_reject': []
            }
        
        best_of_size = {} # each round
        all_history = []         # 所有候選自動機的紀錄
    
        # find best result using beam search
        while True:
            print("======================================")
            print("Beam Search Iteration:", self.iteration)
            # if self.iteration > 100:
            #     print("Max iterations reached. Stopping.")
            #     break

            # create new candidate anchors by adding features to current best anchors
            automatas = AUTO_INSTANCE.propose_automata(self.automatas, self.state, self.sample_fcn, self.iteration, best_of_size.get(self.iteration, []), self.type, beam_size)

            # if no better coverage found with added features -> break
            if len(automatas) == 0:
                print("No candidates survived. Stopping.")
                break

            # for each result, get initial nb of samples used and acc(A)
            stats = self.get_init_stats(automatas)

            # apply KL-LUCB and return result options (nb of options = beam width) in the form of indices
            candidate_automatas = self.kllucb_automata(
                automatas,
                stats,
                epsilon,
                delta,
                batch_size,
                min(beam_size, len(automatas)),
                verbose=verbose,
                verbose_every=verbose_every,
            )
            # store best automatas for the given result size (nb of features in the result)
            best_of_size[self.iteration+1] = [automatas[index] for index in candidate_automatas]
            beam = best_of_size[self.iteration+1]
            # for each candidate result:
            #   update accuracy, lower and upper bounds until accuracy constraints are met
            stats = self.get_init_stats(best_of_size[self.iteration+1], coverages=True)
            positives, negatives, n_accepted, n_samples = stats['positives'], stats['negatives'], stats['n_accepted'], stats['n_samples']
            beta = np.log(1. / (delta / (1 + (beam_size - 1) * len(automatas))))
            kl_constraints = beta / n_samples
            denom = positives + negatives
            means = np.divide(
                denom,
                n_samples,
                out=np.zeros_like(n_samples),
                where=n_samples != 0
            )
            lbs = self.dlow_bernoulli(means, kl_constraints)
            ubs = self.dup_bernoulli(means, kl_constraints)
            if verbose:
                # print('Best of size ', self.iteration, ':')
                for i, mean, lb, ub in zip(candidate_automatas, means, lbs, ubs):
                    print(f"Candidate Automata ID: {id(automatas[i])}, 'mean': {mean}, 'lb': {lb}, 'ub': {ub}")

            # draw samples to ensure result meets accuracy criteria
            continue_sampling = self.to_sample(means, ubs, lbs, accuracy_threshold, epsilon_stop)
            while continue_sampling.any():
                selected_automatas = [automatas[idx] for idx in candidate_automatas[continue_sampling]]
                # selected_dfas = [beam[idx] for idx in continue_sampling]
                true_accept, false_reject, total, accepted = self.draw_samples(selected_automatas, batch_size)
                positives[continue_sampling] += true_accept
                negatives[continue_sampling] += false_reject
                n_samples[continue_sampling] += total
                n_accepted[continue_sampling] += accepted
                denom = positives[continue_sampling] + negatives[continue_sampling]
                means[continue_sampling] = np.divide(
                    denom,
                    n_samples[continue_sampling],
                    out=np.zeros_like(n_samples[continue_sampling]),
                    where=n_samples[continue_sampling] != 0
                )
                kl_constraints[continue_sampling] = beta / n_samples[continue_sampling]
                lbs[continue_sampling] = self.dlow_bernoulli(
                    means[continue_sampling],
                    kl_constraints[continue_sampling],
                )
                ubs[continue_sampling] = self.dup_bernoulli(
                    means[continue_sampling],
                    kl_constraints[continue_sampling],
                )

                continue_sampling = self.to_sample(means, ubs, lbs, accuracy_threshold, epsilon_stop)
            
            # Collect full history
            for automata, m, lb in zip(beam, means, lbs):
                states = len(automata.states)

                # Skip candidates with no reachable accepting state (invalid DFA)
                if automaton_type == "DFA" and not any(s.is_accepting for s in automata.states):
                    print(f"  [SKIP] Candidate {id(automata)} has no reachable accepting state, excluding from history.")
                    continue

                # 最少需要2個狀態，避免trivial單狀態自動機
                min_states = 2
                if states < min_states:
                    continue

                record = {
                    "automata": automata,
                    "accuracy": float(m),
                    "lb": float(lb),
                    "states": states,
                }
                all_history.append(record)

            if verbose:
                for i, mean, lb, ub in zip(candidate_automatas, means, lbs, ubs):
                    t = id(automatas[i])
                    print('%s training accuracy = %.2f lb = %.2f ub = %.2f n: %d' %
                        (t, float(mean), float(lb), float(ub), int(self.state['t_nsamples'][t])))
            self.iteration += 1

        # plot beam search statistics
        iteration_stats = []
        for i in range(self.iteration + 1):
            if i + 1 not in best_of_size:
                continue
            automatas = best_of_size[i + 1]
            iteration_stats.append({
                "iteration": i,
                "training_accuracies": [self.get_automata_metadata(d, success=True)["training_accuracy"] for d in automatas],
                "testing_accuracies": [self.get_automata_metadata(d, success=True)["testing_accuracy"] for d in automatas],
                "states": [len(d.states) for d in automatas],
            })

        if iteration_stats:
            print("Initial training accuracy : ",iteration_stats[0]["training_accuracies"])
            print("Initial testing accuracy : ",iteration_stats[0]["testing_accuracies"])
            print("Initial number of states : ",iteration_stats[0]["states"])
            if(automaton_type == "DFA"):
                AUTO_INSTANCE.automaton_to_graphviz(origin_automata, output_dir=output_dir)
            if automaton_type == "RA":
                AUTO_INSTANCE.automaton_to_graphviz(origin_automata, filename="origin_automata", output_dir=output_dir)

            plot_dfa_beam_stats(iteration_stats, beam_size, output_dir=output_dir)

        # ====== 根據使用者指定的 select_by 決定回傳策略 ======
        def _cleanup_and_return(best_record, success, label=""):
            """統一的 cleanup + return 邏輯"""
            automata = best_record["automata"]
            if automaton_type == "RA":
                print(f"\n[FINAL CLEANUP] Cleaning up {label} RA before returning...")
                AUTO_INSTANCE._remove_unreachable_states(automata, verbose=True)
                for state_node in list(automata.states):
                    if state_node in automata.transitions:
                        AUTO_INSTANCE._dedup_outgoing(automata, state_node, verbose=True)
                AUTO_INSTANCE.automaton_to_graphviz(automata, filename="final_automata", output_dir=output_dir)
            elif automaton_type == "DFA":
                print(f"\n[FINAL CLEANUP] Cleaning up {label} DFA before returning...")
                from learner.dfa_learner import remove_unreachable_states
                remove_unreachable_states(automata)
                n_states_after = len(automata.states)
                has_accept = any(s.is_accepting for s in automata.states)
                print(f"  States after cleanup: {n_states_after}, has accepting state: {has_accept}")
                AUTO_INSTANCE.automaton_to_graphviz(automata, output_dir=output_dir)
            return self.get_automata_metadata(automata, success=success, batch_size=batch_size)

        print(f"\n[SELECT] select_by='{select_by}', accuracy_threshold={accuracy_threshold}, state_threshold={state_threshold}")
        print(f"  Total candidates in history: {len(all_history)}")

        if all_history:
            if select_by == "accuracy":
                # ===== 模式 1: 以 accuracy_threshold 為主 =====
                # 找精確度 >= accuracy_threshold 的候選，從中選狀態數最小的
                qualified = [r for r in all_history if r["accuracy"] >= accuracy_threshold]
                if qualified:
                    best = min(qualified, key=lambda x: x["states"])
                    print(f"  [accuracy mode] {len(qualified)} candidates meet accuracy >= {accuracy_threshold}.")
                    print(f"  Selected: states={best['states']}, accuracy={best['accuracy']:.4f}")
                    return _cleanup_and_return(best, success=True, label="qualified(accuracy)")
                else:
                    # 沒有任何候選達到 accuracy_threshold → 回傳精確度最高的 (best-effort)
                    best = max(all_history, key=lambda x: x["accuracy"])
                    print(f"  [accuracy mode] No candidate meets accuracy >= {accuracy_threshold}.")
                    print(f"  Best-effort: states={best['states']}, accuracy={best['accuracy']:.4f}")
                    return _cleanup_and_return(best, success=False, label="best-effort(accuracy)")

            elif select_by == "state":
                # ===== 模式 2: 以 state_threshold 為主 =====
                # 找狀態數 <= state_threshold 的候選，從中選精確度最高的
                under_state = [r for r in all_history if r["states"] <= state_threshold]
                if under_state:
                    best = max(under_state, key=lambda x: x["accuracy"])
                    print(f"  [state mode] {len(under_state)} candidates have states <= {state_threshold}.")
                    print(f"  Selected: states={best['states']}, accuracy={best['accuracy']:.4f}")
                    return _cleanup_and_return(best, success=True, label="qualified(state)")
                else:
                    # 沒有任何候選的狀態數 <= state_threshold → 回傳精確度最高的 (best-effort)
                    best = max(all_history, key=lambda x: x["accuracy"])
                    print(f"  [state mode] No candidate has states <= {state_threshold}.")
                    print(f"  Best-effort: states={best['states']}, accuracy={best['accuracy']:.4f}")
                    return _cleanup_and_return(best, success=False, label="best-effort(state)")
            else:
                raise ValueError(f"Unknown select_by='{select_by}'. Use 'accuracy' or 'state'.")

        # Safety fallback: 沒有任何候選產生，回傳初始自動機
        print("\n[SELECT] No candidates generated during beam search. Returning initial automaton.")
        if automaton_type == "RA":
            print("\n[FINAL CLEANUP] Cleaning up fallback RA before returning...")
            AUTO_INSTANCE._remove_unreachable_states(origin_automata, verbose=True)
            for state in list(origin_automata.states):
                if state in origin_automata.transitions:
                    AUTO_INSTANCE._dedup_outgoing(origin_automata, state, verbose=True)
        return self.get_automata_metadata(origin_automata, success=False, batch_size=batch_size)