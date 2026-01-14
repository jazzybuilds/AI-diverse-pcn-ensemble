import sys
import pathlib
import matplotlib.pyplot as plt
import torch
sys.path.append(str(pathlib.Path(__file__).parent.parent))
#sys.path.append(str(pathlib.Path().resolve().parent))
from pypc import utils
from pypc import datasets
from pypc import optim
from pypc.models import PCModel



def main(cf):
    print(f"\nStarting discrimnative experiment --seed {cf.seed} --device {utils.DEVICE}")
    utils.seed(cf.seed)

    train_dataset = datasets.MNIST(
        train=True, scale=cf.label_scale, size=cf.train_size, normalize=cf.normalize
    )
    test_dataset = datasets.MNIST(
        train=False, scale=cf.label_scale, size=cf.test_size, normalize=cf.normalize
    )
    train_loader = datasets.get_dataloader(train_dataset, cf.batch_size)
    test_loader = datasets.get_dataloader(test_dataset, cf.test_size)#cf.batch_size)

    model = PCModel(
        nodes=cf.nodes,
        mu_dt=cf.mu_dt,
        act_fn=cf.act_fn,
        use_bias=cf.use_bias,
        kaiming_init=cf.kaiming_init,
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
        metrics = {"acc": []}
        for epoch in range(1, cf.n_epochs + 1):

            #print(f"\nTrain @ epoch {epoch} ({len(train_loader)} batches)")
            for batch_id, (img_batch, label_batch) in enumerate(train_loader):
                model.train_batch_supervised(
                    img_batch, label_batch, cf.n_train_iters, fixed_preds=cf.fixed_preds_train
                )

                optimizer.step(
                    curr_epoch=epoch,
                    curr_batch=batch_id,
                    n_batches=len(train_loader),
                    batch_size=img_batch.size(0),
                )
            if epoch % cf.test_every == 0:
                acc = 0
                for _, (img_batch, label_batch) in enumerate(test_loader):
                    label_preds = model.test_batch_supervised(img_batch)
                    acc += datasets.accuracy(label_preds, label_batch)
                metrics["acc"].append(acc / len(test_loader))
                print("Test @ epoch {} / Accuracy: {:.2%}".format(epoch, acc / len(test_loader)))


    plt.figure()
    plt.plot(metrics["acc"])
    plt.title("PCN Test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

if __name__ == "__main__":
    cf = utils.AttrDict()
    cf.seeds = [1]

    for seed in cf.seeds:

        # experiment params
        cf.seed = seed
        cf.n_epochs = 10
        cf.test_every = 1

        # dataset params
        cf.train_size = None
        cf.test_size = 10000
        cf.label_scale = None
        cf.normalize = False

        # optim params
        cf.optim ="Adam" # "SGD" # 
        cf.lr = 1e-3
        cf.batch_size = 640
        cf.batch_scale = False
        cf.grad_clip = 50
        cf.weight_decay = None

        # inference params
        cf.mu_dt = 0.01
        cf.n_train_iters = 200
        cf.fixed_preds_train =  False

        # model params
        cf.use_bias = True
        cf.kaiming_init = False
        cf.nodes = [784, 300, 100, 10] # mnist and fashiomnist  28*28
        cf.act_fn = utils.Tanh()

        main(cf)
