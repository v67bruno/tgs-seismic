import matplotlib.pyplot as plt

def dataview(input, label, prediction) :
    
    fig = plt.figure(figsize=(15, 15))

    grid = plt.GridSpec(1, 3)
    grid.update(bottom=0.4)
    VV_ax = fig.add_subplot(grid[0, 0])
    VH_ax = fig.add_subplot(grid[0, 1])
    WIND_ax = fig.add_subplot(grid[0, 2])

    VV_ax.imshow(np.log(np.abs(input[0]))*10)
    VV_ax.set_title("VV")
    VH_ax.imshow(np.log(np.abs(input[1]))*10)
    VH_ax.set_title("VH")
    WIND_ax.imshow(input[2])
    WIND_ax.set_xlabel("WIND")

    grid = plt.GridSpec(1, 2)
    grid.update(top=0.4)
    TRUE_ax = fig.add_subplot(grid[0, 0])
    LABEL_ax = fig.add_subplot(grid[0, 1])
    
    TRUE_ax.imshow(label)
    TRUE_ax.set_xlabel("TRUE")
    LABEL_ax.imshow(prediction)
    LABEL_ax.set_xlabel("PREDICTED")

    plt.show()

#################################################################
#################################################################
#################################################################

def gen_pred(train_loader, model) :

    #data_iter = lambda data : next(data)
    data_in = lambda data : data[..., 256:-256, 256:-256]
    data_out = lambda model, data : model(data)

    for idx, data in enumerate(train_loader, 0) :
    
        input, label = data
        input, label = data_in(input), data_in(label)
        pred = data_out(model, input)
        
        for idx in range(len(input)) :
            dataview(input[idx], label[idx], pred[idx].argmax(dim=0))

#################################################################
#################################################################
#################################################################