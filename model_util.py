class newModel1(nn.Module):
    def __init__(self, vocab_size=26):
        super().__init__()
        self.hidden_dim = 256
        self.batch_size = 256
        self.emb_dim = Emb_dim
        
        # self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=2)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        # self.gmlp_t=gMLP(num_tokens = 1000,dim = 32, depth = 2,  seq_len = 40, act = nn.Tanh())
        self.gru = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=6, 
                               bidirectional=True, dropout=0.05)
        
        
        self.block1=nn.Sequential(nn.Linear(3584,1024),
                                            nn.BatchNorm1d(1024),
                                            nn.LeakyReLU(),
                                            nn.Linear(1024,512),
                                            nn.BatchNorm1d(512),
                                            nn.LeakyReLU(),
                                            nn.Linear(512,256),
                                 )

        self.block2=nn.Sequential(
                                               nn.BatchNorm1d(256),
                                               nn.LeakyReLU(),
                                               nn.Linear(256,128),
                                               nn.BatchNorm1d(128),
                                               nn.LeakyReLU(),
                                               nn.Linear(128,64),
                                               nn.BatchNorm1d(64),
                                               nn.LeakyReLU(),
                                               nn.Linear(64,2)
                                            )
        
    def forward(self, x):
        # x=self.embedding(x)
        # output=self.transformer_encoder(x).permute(1, 0, 2)
        # output=self.gmlp_t(x).permute(1, 0, 2)
        x=x.view(1,x.shape[0],x.shape[1])
        # output=self.gmlp_t(x).permute(1, 0, 2)
        # print(output.shape)
        output,hn=self.gru(x)
        output=output.permute(1,0,2)
        hn=hn.permute(1,0,2)
        output=output.reshape(output.shape[0],-1)
        hn=hn.reshape(output.shape[0],-1)
        output=torch.cat([output,hn],1)
        # print('output.shape',output.shape)
        output=self.block1(output)
        return self.block2(output)
class newModel2(nn.Module):
    def __init__(self, vocab_size=26):
        super().__init__()
        self.hidden_dim = 48
        self.batch_size = 256
        self.emb_dim = Emb_dim
        
        # self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=2)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        # self.gmlp_t=gMLP(num_tokens = 1000,dim = 32, depth = 2,  seq_len = 40, act = nn.Tanh())
        # self.gru = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=4, 
        #                        bidirectional=True, dropout=0.2)
        self.c1_1 = nn.Conv1d(32, 256, 1)
        self.c1_2 = nn.Conv1d(32, 256, 3)
        self.c1_3 = nn.Conv1d(32, 256, 5)
        self.p1 = nn.MaxPool1d(3, stride=3)     
        self.c2 = nn.Conv1d(256, 128, 3)
        self.p2 = nn.MaxPool1d(3, stride=3)  
        self.c3 = nn.Conv1d(128, 128, 3)
        # self.p3 = nn.MaxPool1d(3, stride=1)       
        self.drop=nn.Dropout(p=0.01)
        self.block2=nn.Sequential(        
                                               nn.Linear(896,512),
                                               nn.BatchNorm1d(512),
                                               nn.LeakyReLU(),
                                               nn.Linear(512,64),
                                               nn.BatchNorm1d(64),
                                               nn.LeakyReLU(),
                                               nn.Linear(64,2)
                                            )
        
    def forward(self, x):
        # x=self.embedding(x)
        # output=self.transformer_encoder(x).permute(1, 0, 2)
        # output=self.gmlp_t(x).permute(1, 0, 2)
        x=x.view(x.shape[0],32,32)
        # x=x.transpose(1,2)
        # output=self.gmlp_t(x).permute(1, 0, 2)
        # print(output.shape)
        c1_1=self.c1_1(x)
        c1_2=self.c1_2(x)
        c1_3=self.c1_3(x)
        c=torch.cat((c1_1, c1_2, c1_3), -1)
        # print(c1_1.shape,c1_2.shape,c1_3.shape,c.shape)
        p = self.p1(c)
        c=self.c2(p)
        p=self.p2(c)
        # print(p.shape)
        c=self.c3(p)
        # print(c.shape)
        # p=self.p3(c)

        # print(p.shape)
        # print('output.shape',output.shape)
        # print(c.shape)
        c=c.view(c.shape[0],-1)
        c=self.drop(c)
        # print(c.shape)
        return self.block2(c)

