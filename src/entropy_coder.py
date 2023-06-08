import constriction
import numpy as np
import torch
import sys


class RangeCoder():
    def __init__(self, means, stds, precision):
        self.mod_precision = 10.0 ** precision
        
        if means is torch.Tensor:
            means = means.numpy()
        if stds is torch.Tensor:
            stds = stds.numpy()
        
        self.means = np.array(means * self.mod_precision, dtype=np.float64)
        self.stds = np.array(stds * self.mod_precision, dtype=np.float64)
        self.model_family = constriction.stream.model.QuantizedGaussian(
                -int(self.mod_precision), int(self.mod_precision))
    
    def encode(self, msg):
        for value in msg:
            if value > 1:
                raise ValueError('Values in msg shoulde be float number in range [-1, 1].')        
        
        message = np.array(msg * self.mod_precision, dtype=np.int32)
        
        encoder = constriction.stream.queue.RangeEncoder()
        encoder.encode(message, self.model_family, self.means, self.stds)
        compressed_msg = encoder.get_compressed()
        
        return compressed_msg
    
    def encode_batch(self, batch):
        if torch.is_tensor(batch):
            if batch.is_cuda:
                batch = batch.cpu()
            batch = batch.detach().numpy()
        
        compressed_msgs = []
        
        for msg in batch:        
            compressed_msgs.append(self.encode(msg))
        
        return compressed_msgs
    
    def decode_batch(self, compressed_msgs):
        decompressed_msgs = []
        
        for compressed_msg in compressed_msgs:
            decompressed_msgs.append(self.decode(compressed_msg))
        
        decompressed_msgs = torch.Tensor(np.array(decompressed_msgs))
        return decompressed_msgs
        
            
    def decode(self, compressed_msg):
        decoder = constriction.stream.queue.RangeDecoder(compressed_msg)
        decompressed_msg = decoder.decode(self.model_family, self.means, self.stds)
        
        return decompressed_msg / self.mod_precision
    
    def put_to_binary(self, msg, path):
        if sys.byteorder != 'little':
            # Let's use the convention that we always save data in little-endian byte order.
            msg.byteswap(inplace=True)
        msg.tofile(path)
    
    def get_from_binary(self, path):
        msg = np.fromfile(path, dtype=np.uint32)
        if sys.byteorder != 'little':
            # Turn data into native byte order before passing it to `constriction`
            msg.byteswap(inplace=True)
        return msg
        


if __name__ == '__main__':
    precision = 6
    
    msg = np.random.rand(9)
    mean = np.mean(msg)
    std = np.std(msg)
    means = np.full((9, ), mean, dtype=np.float64)
    stds = np.full((9, ), std, dtype=np.float64)
    
    coder = RangeCoder(means, stds, precision)
    
    print(msg)
    compressed_msg = coder.encode(msg)
    print(compressed_msg)
    decompressed_msg = coder.decode(compressed_msg)
    print(decompressed_msg)
