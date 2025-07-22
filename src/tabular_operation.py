import numpy as np

def decode_to_raw_without_bin(sampler, decode_sample):
    """ 
    將離散樣本 ([30,1,0.....]) 轉成原始特徵 (["age =30", "education = Bachelors", ...])
    """
    feature_names = sampler.feature_names
    feature_values = sampler.feature_values
    categorical_features = sampler.categorical_features # 原始特徵中類別特徵的位置
    feature_interval = sampler.disc.feature_intervals # 原始特徵中離散數值特徵的位置

    decoded = []
    for idx, value in enumerate(decode_sample):
        name = feature_names[idx]
        # 類別型
        if idx in categorical_features:
            label = feature_values[idx][value]
            decoded.append(f"{name} : {label}")
        # 未分 bin 的數值型
        elif idx in feature_interval:
            decoded.append(f"{name} : {value}")
        # 其他
        else:
            label = feature_values[idx][value]
            decoded.append(f"{name} : {label}")
    return decoded

def decode_to_raw_with_bin(sampler, decode_samples):
    """ 
    將離散樣本 ([30,1,0.....]) 轉成原始特徵 (["age <= 28", "education = Bachelors", ...])
    """
    bin_samples = sampler.disc.discretize(decode_samples) # decode to bin
    feature_names = sampler.feature_names
    feature_values = sampler.feature_values
    decoded = []
    for idx, value in enumerate(bin_samples):
        name = feature_names[idx]
        label = feature_values[idx][value]
        # 類別型
        decoded.append(f"{name} : {label}")
    return decoded

def binary_to_raw_without_bin(explainer, binary_sample):
    """
    將二元向量轉成原始特徵(["age = 30", "education = Bachelors", ...])
    """
    coverage_data = explainer.mab.state['coverage_data'] 
    matches = np.where((coverage_data == binary_sample).all(axis=1))[0]
    if matches.size == 0:
        raise ValueError("找不到二元遮罩")
    idx = matches[0]
    return explainer.samplers[0].raw_coverage_data[idx]

def binary_to_raw_with_bin(mab, binary_sample):
    """
    把二元向量轉成原始特徵 (with bin) (["age <= 28", "education = Bachelors", ...])
    """
    decoded = []
    seen_feature = set()
    for i, bin_idx in enumerate(binary_sample):
        # bin_idx = None
        feature_idx = mab.sample_fcn.enc2feat_idx[i]
        if feature_idx in seen_feature:
            continue  # 跳過重複的 encoding
        seen_feature.add(feature_idx)
        # 類別型
        if i in mab.sample_fcn.cat_lookup: 
            bin_idx = mab.sample_fcn.cat_lookup[i]
            label = mab.sample_fcn.feature_values[feature_idx][bin_idx]
            decoded.append(f"{mab.sample_fcn.feature_names[feature_idx]} : {label}")
        # 經離散化的數值型
        elif i in mab.sample_fcn.ord_lookup:
            bin_idx = list(mab.sample_fcn.ord_lookup[i])[0]
            interval = mab.sample_fcn.feature_values[feature_idx][bin_idx]
            decoded.append(f"{interval}")
        # 其他
        else:
            feature_idx = mab.sample_fcn.enc2feat_idx[i]
            decoded.append(f"{mab.sample_fcn.feature_names[feature_idx]} : {bin_idx}")
    return decoded