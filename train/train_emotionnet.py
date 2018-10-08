#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/8 14:46
# @Author  : NYY
# @Site    : www.niuyuanyuanna.git.io
# @File    : train_emotionnet.py
import numpy as np

from model.keras_models import make_resnet


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Confusion():
    def __init__(self, loader):
        self.classes = loader.dataset.classes
        self.class_to_idx = loader.dataset.class_to_idx
        self.count = np.zeros((len(self.classes), len(self.classes)))
        # Rows are inputs, cols are outputs

    def add(self, output, input):
        output = output.cpu().data.numpy()
        input = input.cpu().numpy()
        for i, o in enumerate(np.argmax(output, axis=1)):
            self.count[input[i], o] += 1

    def print_confusion(self):
        out = np.copy(self.count)
        col_sums = out.sum(axis=1)
        out = out / col_sums[:, np.newaxis]
        print(out)

    def print_latex(self, avg):
        out = np.copy(self.count)
        col_sums = out.sum(axis=1)
        out = out / col_sums[:, np.newaxis]
        # Print LaTeX style
        print('\\begin{tabular}{' + 'l'*(len(self.classes)+1) + '}')
        print('\\hline')
        print('\t & ', end='')
        print(*self.classes, sep=' & ', end='\\\\\n')
        print('\\hline')
        for ri, r in enumerate(out):
            print('\t{} & '.format(self.classes[ri]), end='')
            for ci, c in enumerate(r):
                print('{0:.3f} {1}'.format(c * 100, '& ' if ci != len(r)-1 else '\\\\'),
                     end=('' if ci != len(r)-1 else '\n'))
        print('\\hline')
        print('\\end{tabular} \\\\')
        print('Average: {:.3}\\%'.format(avg))


class EmotionNet():
    def __init__(self):
        num_classes = 7
        self.model = make_resnet()
        self.model.cuda()
        self.bestaccur = 0.0

    def save_model(self, is_best, filename='checkpoint.pth.tar'):
        torch.save(self.model.state_dict(), filename)
        if is_best:
            shutil.copyfile(filename, 'best-' + filename)

    def load_checkpoint(self, filename):
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(filename))
        else:
            self.model.load_state_dict(torch.load(filename, lambda storage, loc: storage))


    def train_model(self, datadir, validdir, outprefix, epochs=10, csvout=None):
        batch_size = 48
        num_workers = 4
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(datadir, transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        valid_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(validdir, transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            criterion = criterion.cuda()

        optimizer = torch.optim.Adam(self.model.parameters())
        # CSV
        if csvout is not None:
            csvf = open(csvout, 'w')

        # Switch to training
        for e in range(epochs):
            self.model.train()
            losses = AverageMeter()
            top1 = AverageMeter()
            # Train
            for i, (input, target) in enumerate(train_loader):
                if torch.cuda.is_available():
                    target = target.cuda(async=True)
                    input_var = Variable(input).cuda()
                else:
                    input_var = Variable(input)
                target_var = Variable(target)
                output = self.model(input_var)

                loss = criterion(output, target_var)
                prec1 = accuracy(output.data, target, topk=(1,))
                losses.update(loss.data[0], input.size(0))
                top1.update(prec1[0][0], input.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 5 == 0:
                    logging.info('Epoch: {} {}/{}\nLoss: {}\nPrec: {}'.format(
                        e, i, len(train_loader), losses.avg, top1.avg))
            # Valid
            lossavg, topavg = self._valid_model(valid_loader)

            logging.info('Epoch: {}, Validation Loss: {}, Prec: {}'.format(
                e, lossavg, topavg))
            if csvout is not None:
                # Epoch, training loss, training accuracy, valid lossavg, topavg
                print('{}, {}, {}, {}, {}'.format(e, losses.avg, top1.avg,
                                                  lossavg, topavg), flush=True, file=csvf)
            top = False
            if self.bestaccur < top1.avg:
                self.bestaccur = top1.avg
                top = True
            self.save_model(top, filename=outprefix + '.pth.tar')
        if csvout is not None:
            csvf.close()

    def valid_model(self, validdir):
        batch_size = 16
        num_workers = 4
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        valid_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(validdir, transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        loss, acc = self._valid_model(valid_loader)
        logging.info('Loss: {}, Precision: {}'.format(loss, acc))

    def classify_one_image(self, imgf,
                           classes=['afraid', 'angry', 'disgusted', 'happy', 'neutral', 'sad', 'surprised']):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transf = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        # Face detection
        args = {}
        args['threshold'] = 0.0
        args['window'] = False
        args['ignore_multi'] = True
        args['grow'] = 10
        args['resize'] = True
        args['row_resize'] = 512
        args['col_resize'] = 512
        args['min_proportion'] = 0.1
        with tempfile.TemporaryDirectory() as tempdir:
            args['o'] = tempdir
            face_detector.transform(extract_faces.AttributeDict(args), [imgf])
            cropped = Image.open(tempdir + '/' + os.path.basename(imgf))
        cropped = transf(cropped)

        input_var = Variable(cropped.view(1, *cropped.shape))

        if torch.cuda.is_available():
            input_var = input_var.cuda()

        output = self.model.forward(input_var).cpu().data.numpy()
        softmax = np.exp(output) / np.sum(np.exp(output))
        clss = np.argmax(softmax)
        fig = plt.figure()
        plt.imshow(Image.open(imgf))
        fig.subplots_adjust(bottom=0.2)
        plt.figtext(0.1, 0.05, ', '.join(classes))
        plt.figtext(0.1, 0.10, ', '.join(['{:.3}'.format(a) for a in softmax.reshape(-1)]))
        plt.title(classes[clss])
        plt.show()

    def _valid_model(self, valid_loader):
        self.model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        confusion = Confusion(valid_loader)
        criterion = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            criterion = criterion.cuda()

        for i, (input, target) in enumerate(valid_loader):
            if torch.cuda.is_available():
                target = target.cuda(async=True)
            input_var = Variable(input, volatile=True)
            if torch.cuda.is_available():
                input_var = input_var.cuda()
            target_var = Variable(target, volatile=True)

            output = self.model(input_var)
            confusion.add(output, target)
            loss = criterion(output, target_var)

            prec1 = accuracy(output.data, target, topk=(1,))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0][0], input.size(0))
        print('Confusion Matrix')
        confusion.print_confusion()
        confusion.print_latex(top1.avg)
        return losses.avg, top1.avg
