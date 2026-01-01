
import torch
from tqdm.auto import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"

# Creating training step function
def train_step(model, dataloader, optimizer, loss_func):
    model.to(device)
    model.train()

    train_loss = 0
    train_acc = 0

    for batch,(images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        loss = loss_func(outputs, labels)

        train_loss = train_loss + loss.item()

        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        predicted = torch.argmax(outputs,dim=1)
        
        train_acc += (predicted == labels).sum().item()/len(predicted)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


# creating evaluate function
def evaluate(model, dataloader, loss_func):
    model.eval()
    test_loss = 0
    test_acc = 0

    with torch.inference_mode():
        for batch,(images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            loss = loss_func(outputs, labels)
            
            test_loss = test_loss + loss.item()
            
            predicted = torch.argmax(outputs, dim=1)
            
            test_acc += (predicted == labels).sum().item()/len(predicted)  # Count correct predictions
    
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


# combine train_step and evaluate to this one function
def train(model, train_dataloader, test_dataloader, optimizer, loss_func, epochs):

    results = {
        "train_loss" : [],
        "train_acc" : [],
        "test_loss" : [],
        "test_acc" : []
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, optimizer, loss_func)

        test_loss, test_acc = evaluate(model, test_dataloader, loss_func)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    return results
