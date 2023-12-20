import torch.nn as nn
import torch
# import torch.nn.functional 
from torchinfo import summary
from tokenizers import Tokenizer


# class CfgNode:
class CN:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CN):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)
    def to_dict(self):
        return { k: v.to_dict() if isinstance(v, CN) else v for k, v in self.__dict__.items() }
    def merge_from_dict(self, d):
        self.__dict__.update(d)

def config():
    C=CN()
    C.DecoderLayers=16
    C.width=256
    C.FFWidth=512
    C.NumberOfHead=8
    C.NumberOfExpert=12
    C.dtype=torch.bfloat16
    C.ContextSizeToken=192
    C.BatchSize=4
    C.VocabSize=52000
    C.device=torch.device("cuda")
    return C
Config=config()




"""
fp=file_process()
fanfic_token=fp.tokenize('all_my_epub_fanfic_collection.txt',vocab_s=52000)
fanfic_tokenizer=fanfic_token
fanfic_vocab=fanfic_tokenizer.get_vocab()
print (sorted(fanfic_vocab.items() ,key=lambda x:x[1]))
with open('fanfic_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(fanfic_vocab, f, ensure_ascii=False)
fanfic_tokenizer.save("fanfic_tokenizer.json")"""


fanfic_tokenizer = Tokenizer.from_file("fanfic_tokenizer.json")

TextInput=["all my live i allways love yukinoshita, haruno, iroha, and yumiko. They are the most precious to me.","i love miu", "the world is ending tomorrow", "don't just sit there"]
# token_a=fanfic_tokenizer.encode_batch(TextInput)
# token_a=fanfic_tokenizer.encode(
# token_a=fanfic_tokenizer(TextInput)
# print (token_a)
# print(token_a[1].ids)
# print (token_a[1].tokens)
# print (token_a[1].attention_mask)

# config={
#     "NumberOfLayer" : 8,
#     "NumberOfHeads" : 8,
#     "HeadWidth" : 128,
#     "NetworkWidth" : 1024,
#     "VocabSize" : 52000,
#     "BatchSize" : 4,
#     "ContextSizeToken" : 2048,
#     "PaddingTokenIndex" : 0,
# }
class FanficBPETokenizer():
    def __init__(self,Config):
        self.Tokenizer = Tokenizer.from_file("fanfic_tokenizer.json")
        self.ContextSize = Config.ContextSizeToken
        self.BatchSize = Config.BatchSize
    def BatchEncode(self, ListOfText):
        token=self.Tokenizer.encode_batch(ListOfText)
        # print (token)
        """
        [84, 1025, 23105]
        [0, 0, 0]
        ['Ġi', 'Ġlove', 'Ġmiu']
        [(0, 1), (3, 6), (8, 10)]
        [1, 1, 1]
        [0, 0, 0]
        []
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        """
        # print (token[1].ids)  #[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing
        # print (token[1].type_ids)
        print (token[1].tokens)
        # print (token[1].offsets)
        # print (token[1].attention_mask)
        # print (token[1].special_tokens_mask)
        # print (token[1].overflowing)
        Token=torch.zeros(self.BatchSize,self.ContextSize)
        # print (Token.shape)
        for n, T in enumerate(token):
            TToken=torch.cat((torch.tensor(T.ids),torch.tensor([2])),0)
            # print (TToken.shape[0])
            # Token[n, 1: TToken.shape[0]+1]=TToken
            Token[n, 0: TToken.shape[0]]=TToken
        # Token[:,0]=1
        TokenLoss=torch.cat((Token[:,1:] , torch.zeros((self.BatchSize,1))), 1)

        # print(Token[:,:30])
        # mask=torch.cat((((Token!=0)*1)[:,1:] , torch.zeros((self.BatchSize,1))), 1)
        # mask=((Token==0)*torch.tensor(float('-inf')))

        # mask=torch.full_like(Token,float('-inf'))
        # mask=mask.masked_fill(TokenLoss != 0, 0)
        mask=(TokenLoss>0)*1

        # [:,1:]
        # mask=torch.cat((mask, torch.zeros((self.BatchSize,1))), 1)

        # print (TokenLoss.shape, Token.shape, mask[:,:30])
        # print (Token[:,:30])
        print (TokenLoss.shape, Token.shape)
        return [Token.to(torch.int32), TokenLoss.to(torch.int32), mask]




class Embedding(nn.Module):
    def __init__(self,Config):
        super().__init__()
        self.TokenEmbedding=nn.Embedding(Config.VocabSize,Config.width)
    def forward(self, x):
        x=self.TokenEmbedding(x)
        return x 
        








class TransformerMoE(nn.Module):
    def __init__(self,Config):
        super().__init__()
        self.EmbeddingLayer=Embedding(Config)
        self.DecoderLayersCount=Config.DecoderLayers
        self.DecoderDictionary=nn.ModuleDict({
            f'{i}': DecoderModule(Config) for i in range (Config.DecoderLayers)
        })
        self.device=Config.device

    def forward(self, X):
        # print (x.get_device())
        x=self.EmbeddingLayer(X[0].to(self.device))
        # print (x.get_device())
        # x=x.to(torch.device("cuda"))
        for n in range(self.DecoderLayersCount):
            # print (x.get_device())
            x=self.DecoderDictionary[str(n)](x)
        # for Decoder in self.DecoderDictionary:
        #     x=Decoder(x)
        return (x)



class DecoderModule(nn.Module):
    def __init__(self,Config):
        super().__init__()
        self.width=Config.width
        self.NumberOfHead=Config.NumberOfHead
        self.dtype=Config.dtype

        self.Q=nn.Linear(self.width,self.width,)
        self.K=nn.Linear(self.width,self.width,)
        self.V=nn.Linear(self.width,self.width,)
        self.LayerNorm=nn.LayerNorm(self.width)
        self.LayerNorm2=nn.LayerNorm(self.width)
        self.MoE=MoE(Config)

        self.device=Config.device



    def forward(self,x):
        batchSize=x.shape[0]
        SeqLength=x.shape[1]
        HeadWidth=int(self.width/self.NumberOfHead)

        Qout=self.Q(x)
        Kout=self.K(x)
        Vout=self.V(x)
        # print (self.width/self.NumberOfHead)
        Qout=torch.reshape(Qout, (batchSize,SeqLength,self.NumberOfHead,HeadWidth))
        Kout=torch.reshape(Kout, (batchSize,SeqLength,self.NumberOfHead,HeadWidth))
        Vout=torch.reshape(Vout, (batchSize,SeqLength,self.NumberOfHead,HeadWidth))
        
        

        QKMaskTemplate=torch.triu(torch.full((SeqLength, SeqLength), float("-inf"), dtype=self.dtype), diagonal=1).to(self.device)

        QKMatrix=torch.einsum('bshw,bShw->bhsS', Qout, Kout)+QKMaskTemplate
        QKMatrix=nn.functional.softmax(QKMatrix,dim=-1)

        QKV=torch.einsum("bhsS,bShw->bshw",QKMatrix,Vout)
        QKV=torch.reshape(QKV, (batchSize, SeqLength, self.width))
        ADD=QKV+x
        # Norm=nn.functional.layer_norm(ADD,self.width)
        Norm=self.LayerNorm(ADD)

        X=self.MoE(Norm)

        X=self.LayerNorm2(X+Norm)

        print (x.shape, X.shape)
        # print (Qout.shape)
        # print (QKMatrix.shape)
        # print (QKMaskTemplate.shape)
        # print (QKMatrix[0,0,0:5,0:5])
        # print (QKV.shape)
        # print (x[0,0,:5])
        # print (X[0,0,:5])
        # print (QKMaskTemplate[0:5,0:5])
        return X
        

class GatedFeedForward(nn.Module):
    def __init__(self,Config):
        super().__init__()
        self.ff1=nn.Linear(Config.width, Config.FFWidth)
        self.ffG=nn.Linear(Config.width, Config.FFWidth)
        self.ff2=nn.Linear(Config.FFWidth, Config.width)
        self.Dropout=nn.Dropout(0.1)
    def forward(self,x):
        x=nn.functional.silu(self.ff1(x))*self.ffG(x)
        x=self.Dropout(x)
        return self.ff2(x)


class MoE(nn.Module):
    def __init__(self, Config):
        super().__init__()
        # self.GFF=GatedFeedForward(Config)
        self.ExpertsGate=nn.Linear(Config.width, Config.NumberOfExpert)
        self.Experts=nn.ModuleDict({f'{i}':GatedFeedForward(Config) for i in range (Config.NumberOfExpert)})

    def forward(self,x):
        ExpertsChoice=self.ExpertsGate(x)
        ExpertsChoice=nn.functional.softmax(ExpertsChoice, dim=-1)
        ExpertsChoice=torch.topk(ExpertsChoice,2,dim=-1)
        # print (ExpertsChoice[1][0,0:5])
        output=torch.zeros_like(x)
        for n, bach in enumerate(ExpertsChoice[1]):
            # print (bach.shape)
            # print (n)
            for m, token in enumerate(bach):
                output[n,m]=self.Experts[str(token[0].item())](x[n,m])+self.Experts[str(token[1].item())](x[n,m])
                # print (token)
        return output
        # return self.Experts["1"](x)
    




device=torch.device("cuda")
# Decoder=DecoderModule(Config).to(device)
# Decoder.forward(torch.randn(32, 24 ,512).to(device))


# MoE1=MoE(Config).to(device)
# summary(MoE1)


TransformerMoE1=TransformerMoE(Config).to(device)
# summary(TransformerMoE1)



FanficTokenizer=FanficBPETokenizer(Config)
BatchOfTokenizedText=FanficTokenizer.BatchEncode(TextInput)



# TransformerMoE1.forward(torch.randn(32, 24 ,256).to(device))
TransformerMoE1.forward(BatchOfTokenizedText)





'''Use microsoft pi, Stack many similar transformer layer, is this what it means GPT4 operating in a loop?'''