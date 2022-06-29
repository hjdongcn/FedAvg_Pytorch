# FedAvg_Pytorch
对WHDY的FedAvg/user_pytorch加了些许注释

运行项目的server.py文件，参数说明如下，参数具有默认值，可直接运行：
```
optional arguments:

  -h, --help            show this help message and exit
  
  -g GPU, --gpu GPU     gpu id to use(e.g. 0,1,2,3) (default: 0)
  
  -nc NUM_OF_CLIENTS, --num_of_clients NUM_OF_CLIENTS
                        numer of the clients (default: 100)
                        
  -cf CFRACTION, --cfraction CFRACTION
                        C fraction, 0 means 1 client, 1 means total clients (default: 0.1)
                        
  -E EPOCH, --epoch EPOCH
                        local train epoch (default: 5)
                        
  -B BATCHSIZE, --batchsize BATCHSIZE
                        local train batch size (default: 10)

  -mn MODEL_NAME, --model_name MODEL_NAME
                        the model to train (default: mnist_cnn)
                        
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate, use value from origin paper as default (default: 0.01)
                        
  -vf VAL_FREQ, --val_freq VAL_FREQ
                        model validation frequency(of communications) (default: 5)
                        
  -sf SAVE_FREQ, --save_freq SAVE_FREQ
                        global model save frequency(of communication) (default: 20)
                        
  -ncomm NUM_COMM, --num_comm NUM_COMM
                        number of communications (default: 1000)
                        
  -sp SAVE_PATH, --save_path SAVE_PATH
                        the saving path of checkpoints (default: ./checkpoints)
                        
  -iid IID, --IID IID   the way to allocate data to clients (default: 0)
  ```
