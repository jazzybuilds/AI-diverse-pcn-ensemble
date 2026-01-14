
import sys
import pathlib
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(pathlib.Path(__file__).parent.parent))
#sys.path.append(str(pathlib.Path().resolve().parent))
from pypc import utils
from pypc import datasets
from pypc import optim
from pypc.models import PCModel


def plot_imgs(imgs, labels, title):
    img_titles=[str(labels[i]) for i in range(len(labels))]
    fig, axes = plt.subplots(2, 8)
    axes = axes.flatten()
    for i, img in enumerate(imgs):
        axes[i].imshow(img, cmap="gray")
        axes[i].set_axis_off()
        if i>=8:
            axes[i].set_title(img_titles[i-8], fontsize=12)
    fig.suptitle(title, fontsize=16) 
    #plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.show()
    plt.close("all")    

def main(cf):
    print(f"\nStarting generative experiment --seed {cf.seed} --device {utils.DEVICE}")
    utils.seed(cf.seed)

    train_dataset = datasets.MNIST(train=True, scale=cf.label_scale, size=cf.train_size, normalize=cf.normalize)
    test_dataset = datasets.MNIST(train=False, scale=cf.label_scale, size=cf.test_size, normalize=cf.normalize)
    train_loader = datasets.get_dataloader(train_dataset, cf.train_batch_size)
    test_loader = datasets.get_dataloader(test_dataset, cf.test_batch_size)

    model = PCModel(
        nodes=cf.nodes, mu_dt=cf.mu_dt, act_fn=cf.act_fn, use_bias=cf.use_bias, kaiming_init=cf.kaiming_init
    )
    optimizer = optim.get_optim(
        model.params,
        cf.optim,
        cf.lr,
        batch_scale=cf.batch_scale,
        grad_clip=cf.grad_clip,
        weight_decay=cf.weight_decay,
    )

    with torch.no_grad():
        metrics = {"acc": [], "train_loss": []}
        
        for epoch in range(1, cf.n_epochs + 1):
            for batch_id, (img_batch, label_batch) in enumerate(train_loader):
                model.train_batch_generative(
                    img_batch, label_batch, cf.n_train_iters, fixed_preds=cf.fixed_preds_train
                )

                optimizer.step(
                    curr_epoch=epoch,
                    curr_batch=batch_id,
                    n_batches=len(train_loader),
                    batch_size=img_batch.size(0),
                )

            acc=0
            if epoch % cf.test_every == 0:
                if cf.increase_test_iters:
                    test_iters=int(cf.n_test_iters*(1+epoch/10)) # gradually increase test_iters with epoch
                    print("Number of test iterations increases with epochs: ", test_iters)
                else:
                    test_iters=cf.n_test_iters
                   
                for _, (img_batch, label_batch) in enumerate(test_loader):                    
                    label_preds = model.test_batch_generative(
                        img_batch, test_iters, fixed_preds=cf.fixed_preds_test,init_std=cf.init_std
                        )            
                    acc += datasets.accuracy(label_preds, label_batch)

                print("Test @ epoch {} / Accuracy: {:.2%}".format(epoch, acc / len(test_loader)))

                if cf.plot_sample_images:    
                    img_preds = model.forward(label_batch)
                    img_preds = img_preds.cpu().detach().numpy()
                    img_preds = img_preds[:8, :]
                    img_true=img_batch.cpu().detach().numpy()[:8,:]
                    img_combined = np.concatenate([img_true, img_preds], axis=0)
                    labels=np.argmax(label_batch.cpu().detach().numpy()[:8,:],axis=1)
                    imgs = [np.reshape(img_combined[i, :], [28, 28]) for i in range(img_combined.shape[0])]
                    plot_imgs(imgs, labels, "Epoch {} Top row is true image, \nbottom row is model generated from True Label".format(epoch))
                    
                    img_preds = model.forward(label_preds)
                    img_preds = img_preds.cpu().detach().numpy()
                    img_preds = img_preds[:8, :]
                    img_true=img_batch.cpu().detach().numpy()[:8,:]
                    img_combined = np.concatenate([img_true, img_preds], axis=0)
                    labels=np.argmax(label_preds.cpu().detach().numpy()[:8,:],axis=1)
                    imgs = [np.reshape(img_combined[i, :], [28, 28]) for i in range(img_combined.shape[0])]
                    plot_imgs(imgs,labels, "Epoch {} Top row is true image, \nbottom row is model generated from Inferred Label".format(epoch))

                    
            metrics["acc"].append(acc / len(test_loader))
           

        plt.figure()
        plt.plot(metrics['acc'])
        plt.title("Accuracy history")
        plt.ylabel("Acccuracy")
        plt.xlabel("Epoch")
        plt.show()

       
        

if __name__ == "__main__":
    cf = utils.AttrDict()
    cf.seeds = [1]

    for seed in cf.seeds:

        # experiment params
        cf.seed = seed
        cf.n_epochs =5
        cf.test_every = 1
        cf.plot_sample_images = False

        # dataset params
        cf.train_size = None
        cf.test_size = 10000
        cf.label_scale = None
        cf.normalize = True

        # optim params
        cf.optim = "Adam"
        cf.lr = 1e-3 
        cf.train_batch_size = 640
        cf.test_batch_size =10000
        cf.batch_scale = False
        cf.grad_clip = None
        cf.weight_decay = None
        
        # inference params
        cf.mu_dt = .01 
        cf.n_train_iters = 100 
        cf.n_test_iters =  500
        cf.increase_test_iters=False # later epochs use more test iterations
        cf.init_std = 0.01 
        cf.fixed_preds_train = False
        cf.fixed_preds_test = False

        # model params
        cf.use_bias = True
        cf.kaiming_init = False
        cf.nodes = [10,100,300,784]
        cf.act_fn = utils.Tanh()

        main(cf)