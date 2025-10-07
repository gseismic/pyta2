import numpy as np
from .utils import IndicatorFunc1Call_XXX

def calculate_weights(n, weights=None):
    """
    计算权重数组
    
    Parameters:
    -----------
    n : int
        窗口长度
    weights : str or np.ndarray or None
        权重类型或自定义权重数组
        - None: 默认使用线性递减权重
        - 'linear': 线性递减权重
        - 'linear_inv': 线性递增权重
        - 'exp': 指数递减权重 (alpha=0.5)
        - 'exp_inv': 指数递增权重 (alpha=0.5)
        - 'exp_{alpha}': 指数递减权重，自定义alpha值
        - 'exp_inv_{alpha}': 指数递增权重，自定义alpha值
        - np.ndarray: 直接使用提供的权重数组
    
    Returns:
    --------
    np.ndarray : 归一化后的权重数组
    """
    if weights is None:
        weights = 'linear'
        
    if isinstance(weights, str):
        if weights == 'linear':
            w = np.arange(1, n + 1)
        elif weights == 'linear_inv':
            w = np.arange(n, 0, -1)
        elif weights.startswith('exp'):
            is_inverse = '_inv' in weights
            # 移除_inv以便解析alpha
            weight_type = weights.replace('_inv', '')
            
            try:
                alpha = float(weight_type.split('_')[1]) if '_' in weight_type else 0.5
                alpha = max(0, min(1, alpha))  # 确保alpha在[0,1]范围内
            except:
                alpha = 0.5
                
            if is_inverse:
                # 指数递增权重
                w = (1 - alpha) ** np.arange(n)
            else:
                # 指数递减权重
                w = (1 - alpha) ** np.arange(n)[::-1]
        else:
            raise ValueError(f"Unsupported weight type: {weights}")
    elif isinstance(weights, np.ndarray):
        if len(weights) != n:
            raise ValueError(f"Weight array length ({len(weights)}) must match window size ({n})")
        w = weights
    else:
        raise ValueError(f"Unsupported weight type: {type(weights)}")
    
    return w / np.sum(w)  # 归一化

class rStd(IndicatorFunc1Call_XXX):
    """
    rolling std
    """
    name = 'Std'
    def __init__(self, n, *args, **kwargs):
        callback = lambda values: np.std(values[-self.n:], axis=-1)
        super(rStd, self).__init__(callback, n, *args, **kwargs)

class rMean(IndicatorFunc1Call_XXX):
    """
    rolling mean
    """
    name = 'Mean'
    def __init__(self, n, *args, **kwargs):
        callback = lambda values: np.mean(values[-self.n:], axis=-1)
        super(rMean, self).__init__(callback, n, *args, **kwargs)

class rMedian(IndicatorFunc1Call_XXX):
    """
    rolling median
    """
    name = 'Median'
    def __init__(self, n, *args, **kwargs):
        callback = lambda values: np.median(values[-self.n:], axis=-1)
        super(rMedian, self).__init__(callback, n, *args, **kwargs)

class rMin(IndicatorFunc1Call_XXX):
    """
    rolling min
    """
    name = 'Min'
    def __init__(self, n, *args, **kwargs):
        callback = lambda values: np.min(values[-self.n:], axis=-1)
        super(rMin, self).__init__(callback, n, *args, **kwargs)

class rMax(IndicatorFunc1Call_XXX):
    """
    rolling max
    """
    name = 'Max'
    def __init__(self, n, *args, **kwargs):
        callback = lambda values: np.max(values[-self.n:], axis=-1)
        super(rMax, self).__init__(callback, n, *args, **kwargs)

class rSum(IndicatorFunc1Call_XXX):
    """
    rolling sum
    """
    name = 'Sum'
    def __init__(self, n, *args, **kwargs):
        callback = lambda values: np.sum(values[-self.n:], axis=-1)
        super(rSum, self).__init__(callback, n, *args, **kwargs)

class rVar(IndicatorFunc1Call_XXX):
    """
    rolling variance (方差)
    """
    name = 'Var'
    def __init__(self, n, *args, **kwargs):
        callback = lambda values: np.var(values[-self.n:], axis=-1)
        super(rVar, self).__init__(callback, n, *args, **kwargs)

class rSkew(IndicatorFunc1Call_XXX):
    """
    rolling skewness (偏度)
    """
    name = 'Skew'
    def __init__(self, n, *args, **kwargs):
        callback = lambda values: np.nanmean(((values[-self.n:] - np.nanmean(values[-self.n:])) / np.nanstd(values[-self.n:]))**3)
        super(rSkew, self).__init__(callback, n, *args, **kwargs)

class rKurt(IndicatorFunc1Call_XXX):
    """
    rolling kurtosis (峰度)
    """
    name = 'Kurt'
    def __init__(self, n, *args, **kwargs):
        callback = lambda values: np.nanmean(((values[-self.n:] - np.nanmean(values[-self.n:])) / np.nanstd(values[-self.n:]))**4)
        super(rKurt, self).__init__(callback, n, *args, **kwargs)

class rQuantile(IndicatorFunc1Call_XXX):
    """
    rolling quantile (分位数)
    """
    name = 'Quantile'
    def __init__(self, n, q=0.5, *args, **kwargs):
        self.q = q
        callback = lambda values: np.quantile(values[-self.n:], q, axis=-1)
        super(rQuantile, self).__init__(callback, n, *args, **kwargs)

class rPercentile(IndicatorFunc1Call_XXX):
    """
    rolling percentile (百分位数)
    """
    name = 'Percentile'
    def __init__(self, n, q=0.5, *args, **kwargs):
        self.q = q
        callback = lambda values: np.percentile(values[-self.n:], q, axis=-1)
        super(rPercentile, self).__init__(callback, n, *args, **kwargs)

class rCumSum(IndicatorFunc1Call_XXX):
    """
    rolling cumulative sum
    """
    name = 'CumSum'
    def __init__(self, n, *args, **kwargs):
        callback = lambda values: np.cumsum(values[-self.n:], axis=-1)
        super(rCumSum, self).__init__(callback, n, *args, **kwargs)
        
class rCumProd(IndicatorFunc1Call_XXX):
    """
    rolling cumulative product
    """
    name = 'CumProd'
    def __init__(self, n, *args, **kwargs):
        callback = lambda values: np.cumprod(values[-self.n:], axis=-1)
        super(rCumProd, self).__init__(callback, n, *args, **kwargs)
        
class rPTP(IndicatorFunc1Call_XXX):
    """
    rolling peak to peak (峰峰值)
    """
    name = 'PTP'
    def __init__(self, n, *args, **kwargs):
        callback = lambda values: np.ptp(values[-self.n:], axis=-1)
        super(rPTP, self).__init__(callback, n, *args, **kwargs)

class rNorm(IndicatorFunc1Call_XXX):
    """
    rolling norm (范数)
    """
    name = 'Norm'
    def __init__(self, n, ord=None, *args, **kwargs):
        self.ord = ord
        callback = lambda values: np.linalg.norm(values[-self.n:], ord=self.ord)
        super(rNorm, self).__init__(callback, n, *args, **kwargs)

class rEntropy(IndicatorFunc1Call_XXX):
    """
    rolling entropy (信息熵)
    """
    name = 'Entropy'
    def __init__(self, n, *args, **kwargs):
        def calc_entropy(values):
            hist, _ = np.histogram(values, bins='auto', density=True)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log(hist))
        callback = lambda values: calc_entropy(values[-self.n:])
        super(rEntropy, self).__init__(callback, n, *args, **kwargs)

class rKurtosisExcess(IndicatorFunc1Call_XXX):
    """
    rolling excess kurtosis (超额峰度)
    """
    name = 'KurtosisExcess'
    def __init__(self, n, *args, **kwargs):
        callback = lambda values: np.nanmean(((values[-self.n:] - np.nanmean(values[-self.n:])) / np.nanstd(values[-self.n:]))**4) - 3
        super(rKurtosisExcess, self).__init__(callback, n, *args, **kwargs)

class rGeoMean(IndicatorFunc1Call_XXX):
    """
    rolling geometric mean (几何平均数)
    """
    name = 'GeoMean'
    def __init__(self, n, *args, **kwargs):
        callback = lambda values: np.exp(np.mean(np.log(np.abs(values[-self.n:]))))
        super(rGeoMean, self).__init__(callback, n, *args, **kwargs)

class rHarmonicMean(IndicatorFunc1Call_XXX):
    """
    rolling harmonic mean (调和平均数)
    """
    name = 'HarmonicMean'
    def __init__(self, n, *args, **kwargs):
        callback = lambda values: 1 / np.mean(1 / values[-self.n:])
        super(rHarmonicMean, self).__init__(callback, n, *args, **kwargs)

class rMode(IndicatorFunc1Call_XXX):
    """
    rolling mode (众数)
    """
    name = 'Mode'
    def __init__(self, n, *args, **kwargs):
        callback = lambda values: float(np.bincount(values[-self.n:].astype(int)).argmax())
        super(rMode, self).__init__(callback, n, *args, **kwargs)

class rSemiVariance(IndicatorFunc1Call_XXX):
    """
    rolling semi-variance (半方差)
    """
    name = 'SemiVariance'
    def __init__(self, n, *args, **kwargs):
        def calc_semi_var(values):
            mean = np.mean(values)
            return np.mean(np.square(np.minimum(values - mean, 0)))
        callback = lambda values: calc_semi_var(values[-self.n:])
        super(rSemiVariance, self).__init__(callback, n, *args, **kwargs)

class rMAD(IndicatorFunc1Call_XXX):
    """
    rolling Median Absolute Deviation (中位数绝对偏差)
    """
    name = 'MAD'
    def __init__(self, n, *args, **kwargs):
        callback = lambda values: np.median(np.abs(values[-self.n:] - np.median(values[-self.n:])))
        super(rMAD, self).__init__(callback, n, *args, **kwargs)

class rRMS(IndicatorFunc1Call_XXX):
    """
    rolling Root Mean Square (均方根)
    """
    name = 'RMS'
    def __init__(self, n, *args, **kwargs):
        callback = lambda values: np.sqrt(np.mean(np.square(values[-self.n:])))
        super(rRMS, self).__init__(callback, n, *args, **kwargs)
        
class rWMean(IndicatorFunc1Call_XXX):
    """
    rolling weighted mean (加权平均数)
    weights参数支持:
    - None或'linear': 线性递减权重
    - 'exp': 指数递减权重(alpha=0.5)
    - 'exp_0.3': 指数递减权重(alpha=0.3)
    - np.ndarray: 自定义权重数组
    """
    name = 'WMean'
    def __init__(self, n, weights=None, *args, **kwargs):
        self.weights = calculate_weights(n, weights)
        callback = lambda values: np.sum(values[-self.n:] * self.weights)
        super(rWMean, self).__init__(callback, n, *args, **kwargs)

class rEWMA(IndicatorFunc1Call_XXX):
    """
    rolling exponentially weighted moving average (指数加权移动平均)
    alpha: 平滑系数 (0,1]，越大表示近期数据权重越高
    """
    name = 'EWMA'
    def __init__(self, n, alpha=0.2, *args, **kwargs):
        self.alpha = alpha
        def calc_ewma(values):
            weights = (1 - self.alpha) ** np.arange(self.n)[::-1]
            weights /= weights.sum()
            return np.sum(values[-self.n:] * weights)
            
        callback = calc_ewma
        super(rEWMA, self).__init__(callback, n, *args, **kwargs)

class rWStd(IndicatorFunc1Call_XXX):
    """
    rolling weighted standard deviation (加权标准差)
    weights参数支持同rWMean
    """
    name = 'WStd'
    def __init__(self, n, weights=None, *args, **kwargs):
        self.weights = calculate_weights(n, weights)
        def calc_wstd(values):
            wmean = np.sum(values[-self.n:] * self.weights)
            return np.sqrt(np.sum(self.weights * (values[-self.n:] - wmean) ** 2))
        callback = calc_wstd
        super(rWStd, self).__init__(callback, n, *args, **kwargs)

class rWVar(IndicatorFunc1Call_XXX):
    """
    rolling weighted variance (加权方差)
    weights参数支持:
    - None或'linear': 线性递减权重
    - 'exp': 指数递减权重(alpha=0.5)
    - 'exp_0.3': 指数递减权重(alpha=0.3)
    - np.ndarray: 自定义权重数组
    """
    name = 'WVar'
    def __init__(self, n, weights=None, *args, **kwargs):
        self.weights = calculate_weights(n, weights)
        def calc_wvar(values):
            wmean = np.sum(values[-self.n:] * self.weights)
            return np.sum(self.weights * (values[-self.n:] - wmean) ** 2)
        callback = calc_wvar
        super(rWVar, self).__init__(callback, n, *args, **kwargs)

class rWSkew(IndicatorFunc1Call_XXX):
    """
    rolling weighted skewness (加权偏度)
    weights参数支持同rWMean
    """
    name = 'WSkew'
    def __init__(self, n, weights=None, *args, **kwargs):
        self.weights = calculate_weights(n, weights)
        def calc_wskew(values):
            wmean = np.sum(values[-self.n:] * self.weights)
            wvar = np.sum(self.weights * (values[-self.n:] - wmean) ** 2)
            wstd = np.sqrt(wvar)
            return np.sum(self.weights * ((values[-self.n:] - wmean) / wstd) ** 3)
        callback = calc_wskew
        super(rWSkew, self).__init__(callback, n, *args, **kwargs)

class rWKurt(IndicatorFunc1Call_XXX):
    """
    rolling weighted kurtosis (加权峰度)
    weights参数支持同rWMean
    """
    name = 'WKurt'
    def __init__(self, n, weights=None, *args, **kwargs):
        self.weights = calculate_weights(n, weights)
        def calc_wkurt(values):
            wmean = np.sum(values[-self.n:] * self.weights)
            wvar = np.sum(self.weights * (values[-self.n:] - wmean) ** 2)
            wstd = np.sqrt(wvar)
            return np.sum(self.weights * ((values[-self.n:] - wmean) / wstd) ** 4)
        callback = calc_wkurt
        super(rWKurt, self).__init__(callback, n, *args, **kwargs)

class rWQuantile(IndicatorFunc1Call_XXX):
    """
    rolling weighted quantile (加权分位数)
    weights参数支持同rWMean
    """
    name = 'WQuantile'
    def __init__(self, n, q=0.5, weights=None, *args, **kwargs):
        self.weights = calculate_weights(n, weights)
        self.q = q
        def weighted_quantile(values):
            sorted_idx = np.argsort(values)
            sorted_weights = self.weights[sorted_idx]
            cumsum_weights = np.cumsum(sorted_weights)
            idx = np.searchsorted(cumsum_weights, self.q)
            return values[sorted_idx[idx]]
        callback = lambda values: weighted_quantile(values[-self.n:])
        super(rWQuantile, self).__init__(callback, n, *args, **kwargs)
        
class rWMAD(IndicatorFunc1Call_XXX):
    """
    rolling weighted Median Absolute Deviation (加权中位数绝对偏差)
    weights参数支持同rWMean
    """
    name = 'WMAD'
    def __init__(self, n, weights=None, *args, **kwargs):
        self.weights = calculate_weights(n, weights)
        def calc_wmad(values):
            wmedian = weighted_quantile(values[-self.n:], 0.5, self.weights)
            deviations = np.abs(values[-self.n:] - wmedian)
            return weighted_quantile(deviations, 0.5, self.weights)
        callback = calc_wmad
        super(rWMAD, self).__init__(callback, n, *args, **kwargs)

class rWRMS(IndicatorFunc1Call_XXX):
    """
    rolling weighted Root Mean Square (加权均方根)
    weights参数支持同rWMean
    """
    name = 'WRMS'
    def __init__(self, n, weights=None, *args, **kwargs):
        self.weights = calculate_weights(n, weights)
        callback = lambda values: np.sqrt(np.sum(self.weights * np.square(values[-self.n:])))
        super(rWRMS, self).__init__(callback, n, *args, **kwargs)

class rWSemiVariance(IndicatorFunc1Call_XXX):
    """
    rolling weighted semi-variance (加权半方差)
    weights参数支持同rWMean
    """
    name = 'WSemiVariance'
    def __init__(self, n, weights=None, *args, **kwargs):
        self.weights = calculate_weights(n, weights)
        def calc_wsemi_var(values):
            wmean = np.sum(values[-self.n:] * self.weights)
            return np.sum(self.weights * np.square(np.minimum(values[-self.n:] - wmean, 0)))
        callback = calc_wsemi_var
        super(rWSemiVariance, self).__init__(callback, n, *args, **kwargs)

class rWGeoMean(IndicatorFunc1Call_XXX):
    """
    rolling weighted geometric mean (加权几何平均数)
    weights参数支持同rWMean
    """
    name = 'WGeoMean'
    def __init__(self, n, weights=None, *args, **kwargs):
        self.weights = calculate_weights(n, weights)
        callback = lambda values: np.exp(np.sum(self.weights * np.log(np.abs(values[-self.n:]))))
        super(rWGeoMean, self).__init__(callback, n, *args, **kwargs)

class rWHarmonicMean(IndicatorFunc1Call_XXX):
    """
    rolling weighted harmonic mean (加权调和平均数)
    weights参数支持同rWMean
    """
    name = 'WHarmonicMean'
    def __init__(self, n, weights=None, *args, **kwargs):
        self.weights = calculate_weights(n, weights)
        callback = lambda values: 1 / np.sum(self.weights / values[-self.n:])
        super(rWHarmonicMean, self).__init__(callback, n, *args, **kwargs)

class rWEntropy(IndicatorFunc1Call_XXX):
    """
    rolling weighted entropy (加权信息熵)
    weights参数支持同rWMean
    """
    name = 'WEntropy'
    def __init__(self, n, weights=None, *args, **kwargs):
        self.weights = calculate_weights(n, weights)
        def calc_wentropy(values):
            hist, _ = np.histogram(values[-self.n:], bins='auto', weights=self.weights, density=True)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log(hist))
        callback = calc_wentropy
        super(rWEntropy, self).__init__(callback, n, *args, **kwargs)

def weighted_quantile(values, q, weights):
    """
    计算加权分位数的辅助函数
    """
    sorted_idx = np.argsort(values)
    sorted_weights = weights[sorted_idx]
    cumsum_weights = np.cumsum(sorted_weights)
    idx = np.searchsorted(cumsum_weights / cumsum_weights[-1], q)
    return values[sorted_idx[idx]]
        
        