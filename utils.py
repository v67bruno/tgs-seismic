import matplotlib.pyplot as plt

def Data_View(input, label, prediction) :
    
    fig = plt.figure(figsize=(15, 15))

    grid = plt.GridSpec(1, 4)
    grid.update(bottom=0.4)
    IMG_ax = fig.add_subplot(grid[0, 0])
    DPT_ax = fig.add_subplot(grid[0, 1])
    TRUE_ax = fig.add_subplot(grid[0, 2])
    LB_ax = fig.add_subplot(grid[0, 3])

    IMG_ax.imshow(input[0])
    IMG_ax.set_title("Img")
    DPT_ax.imshow(input[1])
    DPT_ax.set_title("Depth")
    TRUE_ax.imshow(label)
    TRUE_ax.set_xlabel("Ground Truth")
    LB_ax.imshow(prediction)
    LB_ax.set_xlabel("PREDICTED")

    plt.show()

#################################################################
#################################################################
#################################################################

def View_Preds(train_loader, model) :

    #data_iter = lambda data : next(data)
    #data_in = lambda data : data[..., 256:-256, 256:-256]
    data_out = lambda model, data : model(data)

    for idx, data in enumerate(train_loader, 0) :
    
        input, label = data
        input, label = input, label
        pred = data_out(model, input)
        
        for idx in range(len(input)) :
            Data_View(input[idx], label[idx], pred[idx].argmax(dim=0))

#################################################################
#################################################################
#################################################################
