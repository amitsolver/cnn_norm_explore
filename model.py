import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
dropout_value=0.01

class Net_Batch(nn.Module):
    def __init__(self) -> None:
        super(Net_Batch,self).__init__()
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout_value)

        #Convolution Block 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=20,kernel_size=(3,3),padding=1,bias=False),
            nn.BatchNorm2d(20)         
        ) # input_size = 32, output_size=32
        
        #Convolution Block 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=20,out_channels=20,kernel_size=(3,3),padding=1,bias=False),
            nn.BatchNorm2d(20)               
        ) # input_size = 32, output_size=32

        #Transition Block 1
        self.transblock1 = nn.Sequential(
            nn.Conv2d(in_channels=20,out_channels=16,kernel_size=(1,1),padding=0,bias=False)
        ) # input_size = 32, output_size=32

        #Pooling Block 1
        self.pool1 = nn.MaxPool2d(2,2) # input_size = 32, output_size=16

        #Convolution Block 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=1,bias=False),
            nn.BatchNorm2d(16)               
        ) # input_size = 16, output_size=16

        #Convolution Block 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=1,bias=False),
            nn.BatchNorm2d(16)               
        ) # input_size = 16, output_size=16

        #Convolution Block 5
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=1,bias=False),
            nn.BatchNorm2d(16)              
        ) # input_size = 16, output_size=16

        #Transition Block 2
        self.transblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=10,kernel_size=(1,1),padding=0,bias=False)
        ) # input_size = 16, output_size=16

        #Pooling Block 2
        self.pool2 = nn.MaxPool2d(2,2) # input_size = 16, output_size=8

        #Convolution Block 6
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10,out_channels=16,kernel_size=(3,3),padding=0,bias=False),
            nn.BatchNorm2d(16)                
        ) # input_size = 8, output_size=6

        #Convolution Block 7
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=0,bias=False),
            nn.BatchNorm2d(16)               
        ) # input_size = 6, output_size=4

        #Convolution Block 8
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=0,bias=False),
            nn.BatchNorm2d(16)             
        ) # input_size = 4, output_size=2
        
        # GAP 1
        self.gap = nn.Sequential(
            nn.AvgPool2d(2)
        ) 

        #Ouput Block
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=10,kernel_size=(1,1),padding=0,bias=False) 
        )

    def forward(self,x):
        x1 = self.dropout(self.Relu(self.convblock1(x)))
        x2 = self.dropout(self.Relu(self.convblock2(x1)))
        x3 = self.transblock1(x2)
        x4 = self.pool1(x3)
        x5 = self.dropout(self.Relu(self.convblock3(x4)))
        x6 = self.dropout(self.Relu(self.convblock4(x5)))
        x7 = self.Relu(self.convblock5(x6)+x5)
        x8 = self.transblock2(x7)
        x9 = self.pool2(x8)
        x10 = self.dropout(self.Relu(self.convblock6(x9)))
        x11 = self.dropout(self.Relu(self.convblock7(x10)))
        x12 = self.dropout(self.Relu(self.convblock8(x11)))
        x13 = self.gap(x12)
        x14 = self.convblock9(x13)
        x15 = x14.view(-1, 10)
        return F.log_softmax(x15, dim=-1)
    

class Net_Layer(nn.Module):
    def __init__(self) -> None:
        super(Net_Layer,self).__init__()
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout_value)

        #Convolution Block 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=20,kernel_size=(3,3),padding=1,bias=False),
            nn.LayerNorm((32,32))         
        ) # input_size = 32, output_size=32
        
        #Convolution Block 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=20,out_channels=20,kernel_size=(3,3),padding=1,bias=False),
            nn.LayerNorm((32,32))              
        ) # input_size = 32, output_size=32

        #Transition Block 1
        self.transblock1 = nn.Sequential(
            nn.Conv2d(in_channels=20,out_channels=16,kernel_size=(1,1),padding=0,bias=False)
        ) # input_size = 32, output_size=32

        #Pooling Block 1
        self.pool1 = nn.MaxPool2d(2,2) # input_size = 32, output_size=16

        #Convolution Block 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=1,bias=False),
            nn.LayerNorm((16,16))                    
        ) # input_size = 16, output_size=16

        #Convolution Block 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=1,bias=False),
            nn.LayerNorm((16,16))              
        ) # input_size = 16, output_size=16

        #Convolution Block 5
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=1,bias=False),
            nn.LayerNorm((16,16))                 
        ) # input_size = 16, output_size=16

        #Transition Block 2
        self.transblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=10,kernel_size=(1,1),padding=0,bias=False)
        ) # input_size = 16, output_size=16

        #Pooling Block 2
        self.pool2 = nn.MaxPool2d(2,2) # input_size = 16, output_size=8

        #Convolution Block 6
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10,out_channels=16,kernel_size=(3,3),padding=0,bias=False),
            nn.LayerNorm((6,6))                  
        ) # input_size = 8, output_size=6

        #Convolution Block 7
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=0,bias=False),
            nn.LayerNorm((4,4))               
        ) # input_size = 6, output_size=4

        #Convolution Block 8
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=0,bias=False),
            nn.LayerNorm((2,2))            
        ) # input_size = 4, output_size=2
        
        # GAP 1
        self.gap = nn.Sequential(
            nn.AvgPool2d(2)
        ) 

        #Ouput Block
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=10,kernel_size=(1,1),padding=0,bias=False) 
        )

    def forward(self,x):
        x1 = self.dropout(self.Relu(self.convblock1(x)))
        x2 = self.dropout(self.Relu(self.convblock2(x1)))
        x3 = self.transblock1(x2)
        x4 = self.pool1(x3)
        x5 = self.dropout(self.Relu(self.convblock3(x4)))
        x6 = self.dropout(self.Relu(self.convblock4(x5)))
        x7 = self.Relu(self.convblock5(x6)+x5)
        x8 = self.transblock2(x7)
        x9 = self.pool2(x8)
        x10 = self.dropout(self.Relu(self.convblock6(x9)))
        x11 = self.dropout(self.Relu(self.convblock7(x10)))
        x12 = self.dropout(self.Relu(self.convblock8(x11)))
        x13 = self.gap(x12)
        x14 = self.convblock9(x13)
        x15 = x14.view(-1, 10)
        return F.log_softmax(x15, dim=-1)



class Net_Group(nn.Module):
    def __init__(self) -> None:
        super(Net_Group,self).__init__()
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout_value)

        #Convolution Block 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=20,kernel_size=(3,3),padding=1,bias=False),
            nn.GroupNorm(2,20)         
        ) # input_size = 32, output_size=32
        
        #Convolution Block 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=20,out_channels=20,kernel_size=(3,3),padding=1,bias=False),
            nn.GroupNorm(2,20)               
        ) # input_size = 32, output_size=32

        #Transition Block 1
        self.transblock1 = nn.Sequential(
            nn.Conv2d(in_channels=20,out_channels=16,kernel_size=(1,1),padding=0,bias=False)
        ) # input_size = 32, output_size=32

        #Pooling Block 1
        self.pool1 = nn.MaxPool2d(2,2) # input_size = 32, output_size=16

        #Convolution Block 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=1,bias=False),
            nn.GroupNorm(2,16)                       
        ) # input_size = 16, output_size=16

        #Convolution Block 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=1,bias=False),
            nn.GroupNorm(2,16)                
        ) # input_size = 16, output_size=16

        #Convolution Block 5
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=1,bias=False),
            nn.GroupNorm(2,16)                   
        ) # input_size = 16, output_size=16

        #Transition Block 2
        self.transblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=10,kernel_size=(1,1),padding=0,bias=False)
        ) # input_size = 16, output_size=16

        #Pooling Block 2
        self.pool2 = nn.MaxPool2d(2,2) # input_size = 16, output_size=8

        #Convolution Block 6
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10,out_channels=16,kernel_size=(3,3),padding=0,bias=False),
            nn.GroupNorm(2,16)                    
        ) # input_size = 8, output_size=6

        #Convolution Block 7
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=0,bias=False),
            nn.GroupNorm(2,16)                  
        ) # input_size = 6, output_size=4

        #Convolution Block 8
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=0,bias=False),
            nn.GroupNorm(2,16)              
        ) # input_size = 4, output_size=2
        
        # GAP 1
        self.gap = nn.Sequential(
            nn.AvgPool2d(2)
        ) 

        #Ouput Block
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=10,kernel_size=(1,1),padding=0,bias=False) 
        )

    def forward(self,x):
        x1 = self.dropout(self.Relu(self.convblock1(x)))
        x2 = self.dropout(self.Relu(self.convblock2(x1)))
        x3 = self.transblock1(x2)
        x4 = self.pool1(x3)
        x5 = self.dropout(self.Relu(self.convblock3(x4)))
        x6 = self.dropout(self.Relu(self.convblock4(x5)))
        x7 = self.Relu(self.convblock5(x6)+x5)
        x8 = self.transblock2(x7)
        x9 = self.pool2(x8)
        x10 = self.dropout(self.Relu(self.convblock6(x9)))
        x11 = self.dropout(self.Relu(self.convblock7(x10)))
        x12 = self.dropout(self.Relu(self.convblock8(x11)))
        x13 = self.gap(x12)
        x14 = self.convblock9(x13)
        x15 = x14.view(-1, 10)
        return F.log_softmax(x15, dim=-1)


    
def model_summary(model,device):
  model = model.to(device)
  summary(model, input_size=(3, 32, 32))
