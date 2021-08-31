import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datasets import Dataset
from torch import optim
from Model import *


dnet = Dnet().cuda()
#dnet.load_state_dict(torch.load("D:/pythonProject1/mydcgan/weight/9-dwight.pkl"))
gnet = Gnet().cuda()
#gnet.load_state_dict(torch.load("D:/pythonProject1/mydcgan/weight/9-gwight.pkl"))
d_opt = optim.Adam(dnet.parameters(), 0.0009, betas=(0.5, 0.999))
g_opt = optim.Adam(gnet.parameters(), 0.0009, betas=(0.5, 0.999))
# d_opt = optim.RMSprop(dnet.parameters(),  0.009)
# g_opt = optim.RMSprop(gnet.parameters(), 0.009)
# scheduler_d = optim.lr_scheduler.ExponentialLR(d_opt, gamma=0.99)
# scheduler_g = optim.lr_scheduler.ExponentialLR(g_opt, gamma=0.99)
Batch_size = 100
loss = torch.nn.BCELoss()
rec_loss = torch.nn.MSELoss()

k = 0

dataset = Dataset("D:/Data_set/Cartoon_faces")
dataloader = DataLoader(dataset, shuffle=True, batch_size=Batch_size)

for epoch in range(10000):
    sum_dloss = 0
    sum_gloss = 0
    dnet.train()
    gnet.train()
    for i, img in enumerate(dataloader):
        img = img.cuda()
        #for j in range(3):
        noise_d = torch.normal(0, 0.02, (img.shape[0], 128, 1, 1), dtype=torch.float32).cuda()
        real_lable = torch.ones([img.shape[0]]).cuda()
        fake_lable = torch.zeros([img.shape[0]]).cuda()
        d_opt.zero_grad()
        real = dnet(img)
        lossd_real = loss(real, real_lable)

        lossd_real.backward(retain_graph=True)
        fake_d = gnet(noise_d)
        fake = dnet(fake_d)
        lossd_fake = loss(fake, fake_lable)

        lossd_fake.backward()
        d_opt.step()
        #scheduler_d.step()


        lossd = lossd_real + lossd_fake
        sum_dloss = lossd.item()

        g_opt.zero_grad()
        #for j in range(3):
        noise_g = torch.normal(0, 0.02, (img.shape[0], 128, 1, 1), dtype=torch.float32).cuda()
        gout = gnet(noise_g)
        #print(gout)
        g2dout = dnet(gout)
        lossg_real_lable = loss(g2dout, real_lable)

        lossg_real_lable.backward(retain_graph=True)
        rec_los = rec_loss(gout, img)
        rec_los.backward()
        # lossg = lossg_real_lable() + rec_los()
        #lossg.backward()
        g_opt.step()
        #scheduler_g.step()
        fake_img = gnet(noise_g)

        lossg = lossg_real_lable + rec_los
        sum_gloss += lossg.item()

        if i % 100 == 0:
            img = img.reshape(-1, 3, 96, 96)
            fake_img = fake_img.reshape(-1, 3, 96, 96)
            save_image(img, "img_face/{}-real_img.jpg".format(k + 1), nrow=10)
            save_image(fake_img, "img_face/{}-fake_img.jpg".format(k + 1), nrow=10)
        k += 1


    avg_dloss = sum_dloss / len(dataloader)
    avg_gloss = sum_gloss / len(dataloader)
    print(epoch, "avg_dloss", avg_dloss)
    print(epoch, "avg_gloss", avg_gloss)

    if epoch % 10 == 9:
        torch.save(dnet.state_dict(), "weight/{}-dwight.pth".format(epoch))
        torch.save(gnet.state_dict(), "weight/{}-gwight.pth".format(epoch))






