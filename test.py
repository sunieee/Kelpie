import torch
import platform
import ctypes

def find_cudnn_version():
    try:
        libcudnn = ctypes.cdll.LoadLibrary('libcudnn.so')
    except OSError:
        return "cuDNN not found"
    
    cudnn_major = ctypes.c_int()
    cudnn_minor = ctypes.c_int()
    cudnn_patch = ctypes.c_int()
    
    libcudnn.cudnnGetVersion.restype = ctypes.c_int
    libcudnn.cudnnGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    libcudnn.cudnnGetVersion(ctypes.byref(cudnn_major), ctypes.byref(cudnn_minor), ctypes.byref(cudnn_patch))
    
    return f"{cudnn_major.value}.{cudnn_minor.value}.{cudnn_patch.value}"

print(f"Python version: {platform.python_version()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {find_cudnn_version()}")


'''
> RuntimeError: GET was unable to find an engine to execute this computation
> https://github.com/microsoft/TaskMatrix/issues/283
降低torch版本，多半是由于torch==2.0.0,使用如下命令（这是我自己解决的方式，仅供参考）：
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
降低torch版本同时把torchvision和torchaudio版本都降低，此外，可以查看自己的torch对应的cuda版本
print(torch.version)
print(torch.version.cuda)
去官网上找匹配的torch安装

> RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling `cublasSgemm
> https://stackoverflow.com/questions/68571902/how-to-fix-runtimeerror-cuda-error-cublas-status-invalid-value-when-calling-cub
pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
'''

class Paths:
    def __init__(self, paths: List[Path], fact_to_explain: Triple) -> None:
        self.paths = paths
        self.fact_to_explain = fact_to_explain
        self.head = [path.head for path in self.paths]
        self.tail = [path.tail for path in self.paths]
        self.triples = []
        for path in self.paths:
            self.triples.extend(path.triples)

    @property
    def score(self):
        return Score(self.head, self.tail, self.triples, self.fact_to_explain)

    @staticmethod
    def from_str(s):
        return Paths([Path(path_str) for path_str in s.split('|')])

    def __len__(self):
        return len(self.paths)

    def __str__(self) -> str:
        return '|'.join([str(path) for path in self.paths])
    

class Score:
    def __init__(self, paths: List[Path], fact_to_explain: Triple, metrics: Dict = {}) -> None:
        self.paths = paths
        self.head = [path.head for path in paths]
        self.tail = [path.tail for path in paths]
        self.fact_to_explain = fact_to_explain
        self.metrics = metrics
        print('init scores:', metrics)

        self.head_exp = Explanation(self.head, fact_to_explain, *self.get_relevance(pt='CA', base='BA'))
        self.tail_exp = Explanation(self.tail, fact_to_explain, *self.get_relevance(pt='AC', base='AB'))
        self.path_exp = Explanation(self.head + self.tail, fact_to_explain, *self.get_relevance(pt='CC', base='BB'))

    def get_retrain_score(self):
        self.head_score = self.head_exp.get_retrain_score()
        self.tail_score = self.tail_exp.get_retrain_score()
        self.path_score = self.path_exp.get_retrain_score()

    def __str__(self):
        return f'''{self.fact_to_explain}: {[str(p) for p in self.paths]}
head: {[str(t) for t in self.head]}: {self.head_exp.relevance}
tail: {[str(t) for t in self.tail]}: {self.tail_exp.relevance}
path: {[str(t) for t in self.head + self.tail]}: {self.path_exp.relevance}
        '''
    
    def get_relevance(self, pt, base):
        try:
            # we want to give higher priority to the facts that, when added, make the score worse (= higher).
            rank_worsening = self.metrics[f'{pt}_rank'] - self.metrics[f'{base}_rank']
            score_worsening = self.metrics[f'{base}_score'] - self.metrics[f'{pt}_score']
            if model.is_minimizer():
                score_worsening = -score_worsening
            # note: the formulation is very different from the addition one
            # return rd(float(rank_worsening + np.tanh(score_worsening)))
            return rd(score_worsening), self.metrics[f'{base}_score'], self.metrics[f'{pt}_score']
        except Exception as e:
            print(e)
            return -1, -1, -1
        
    def has_negative(self):
        return self.head_exp.relevance < 0 or self.tail_exp.relevance < 0 or self.path_exp.relevance < 0
    
    def path_negative(self):
        return self.path_exp.relevance < 0

    @property
    def max_relevance(self):
        return max(self.head_exp.relevance, self.tail_exp.relevance, self.path_exp.relevance)
    
class Explanation:
    def __init__(self, triples: List[Triple], sample_to_explain: Triple, relevance, base, pt) -> None:
        self.triples = triples
        self.sample_to_explain = sample_to_explain
        self.relevance = relevance
        self.base = base
        self.pt = pt
        print('init explanation', [str(t) for t in self.triples], self.relevance, self.base, self.pt)


    def __str__(self) -> str:
        return str(self.triples)
    
    def __len__(self):
        return len(self.triples)

    def get_retrain_score(self) -> Score:
        return self.sample_to_explain.origin_score(self.triples)