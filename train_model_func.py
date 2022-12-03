import tqdm
def train_model(model, optimizer, train_dl, epochs=3, train_func=None, test_func=None, 
                scheduler=None, save_file=None, accelerator=None, epoch_len=None):  # accelerator：适用于多卡的机器，epoch_len到该epoch提前停止
    best_f1 = -1
    for epoch in range(epochs):
        model.train()
        print(f'\nEpoch {epoch+1} / {epochs}:')
        if accelerator:
            pbar = tqdm(train_dl, total=epoch_len, disable=not accelerator.is_local_main_process)
        else: 
            pbar = tqdm(train_dl, total=epoch_len)
        metricsums = {}
        iters, accloss = 0, 0
        for ditem in pbar:
            metrics = {}
            loss = train_func(model, ditem)
            if type(loss) is type({}):
                metrics = {k:v.detach().mean().item() for k,v in loss.items() if k != 'loss'}
                loss = loss['loss']
            iters += 1; accloss += loss
            optimizer.zero_grad()
            if accelerator: 
                accelerator.backward(loss)
            else: 
                loss.backward()
            optimizer.step()
            if scheduler:
                if accelerator is None or not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
            for k, v in metrics.items(): metricsums[k] = metricsums.get(k,0) + v
            infos = {'loss': f'{accloss/iters:.4f}'}
            for k, v in metricsums.items(): infos[k] = f'{v/iters:.4f}' 
            pbar.set_postfix(infos)
            if epoch_len and iters > epoch_len: break
        pbar.close()
        if test_func:
            if accelerator is None or accelerator.is_local_main_process: 
                model.eval()
                f1 = test_func()
                if f1 >best_f1 and save_file:
                    if accelerator:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        accelerator.save(unwrapped_model.state_dict(), save_file)
                    else:
                        torch.save(model.state_dict(), save_file)
                    print(f"Epoch {epoch + 1}, best model saved. (f1={f1:.4f})")
                    best_f1 = f1
def test_func(): 
    global val_time
    vt1 = time.time()
    yt, yp = [], []
    model.eval()
    with torch.no_grad():
        for xx, yy in testloader:
            #zz = model(xx).detach().cpu().argmax(-1)
            zz = (model(xx.to(device))>0.5).long().detach().cpu()
            yp.append(zz)
            yt.append(yy)
    yp = torch.cat(yp,0)
    yt = torch.cat(yt,0)
    accu = (yp == yt).float().mean()
    prec = (yp + yt >1.5).float().sum() / max(yp.sum().item(),1)
    reca = (yp + yt >1.5).float().sum() / max(yt.sum().item(),1)
    f1 = 2*prec*reca/(prec+reca)
    f1 = 0 if isnan(f1) else f1
    record['val_f1'].append(f1)
    #accu = (np.array(yt) == np.array(yp)).sum() / len(yp)
    stri = f'Accuracy: {accu:.4f},  Precision: {prec:.4f},  Recall: {reca:.4f},  F1: {f1:.4f}'
    print(stri)
    vt2 = time.time()
    val_time += vt2-vt1
    model.train()
    return f1
