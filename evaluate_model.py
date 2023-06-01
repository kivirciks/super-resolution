def evaluate(model_name):
    with open("../metrics.txt", "r") as version_file:
        version = int(version_file.readline()) + 1
    if model_name not in models.keys():
        raise Exception("Model with name " + model_name + " is not found")
    connection = connect_database()
    model_id = check_model(connection, model_name, version)
    if model_id is None:
        raise Exception("Model with name " + model_name + " and version " + str(version) + " is not exists")

    download_model(model_name)
    params = {}
    filename = "model" + models[model_name][2]
    if models[model_name][1] == "tensorflow":
        params["input_length"] = x_test.shape[1]
    model = models[model_name][0](params, os.path.join(".", "loaded", filename))
    start = time.time()
    preds = model.predict(x_test)
    end = time.time()
    duration = end - start
    if models[model_name][1] == "tensorflow":
        preds = np.array([np.argmax(i) for i in preds])
    accuracy = calculate_accuracy(preds, y_test)
    
    if args.model == 'sub':
        coef = 10
    elif args.model == 'srcnn':
        coef = 9
    elif args.model == 'vdsr':
        coef = 1
    elif args.model == 'edsr':
        coef = 4
    elif args.model == 'fsrcnn':
        coef = 2
    elif args.model == 'srgan':
        coef = 7
    else:
        raise Exception("the model does not exist")
    for i in model:
    final = ((PNSR * 0.5) +  (Color * 0.3) + (Black * 0.2)) * coef
    
    # сортировка
    a = []
    for i in range(final):
    print(a)
    for i in range(final-1):
        for j in range(final-i-1):
            if a[j] > a[j+1]:
                a[j], a[j+1] = a[j+1], a[j]
    print(a)
