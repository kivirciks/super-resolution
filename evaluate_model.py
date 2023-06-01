def evaluate(model_name):
    with open("../metrics.txt", "r") as version_file:
        version = int(version_file.readline()) + 1
    if model_name not in models.keys():
        raise Exception("Model with name " + model_name + " is not found")
    connection = connect_database()
    model_id = check_model(connection, model_name, version)
    if model_id is None:
        raise Exception("Model with name " + model_name + " and version " + str(version) + " is not exists in "
                                                                                           "database")
    df = download_dataset(connection, "dataset_test", int_categories=True)
    x_test, y_test = tokenize_dataset(df)

    download_model(connection, model_name, "loaded.zip", version)
    os.makedirs("loaded")
    shutil.unpack_archive("loaded.zip", "loaded", "zip")
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
        
    upload_metrics(connection, accuracy, duration, model_id)

    connection.close()
for i in model:
  final = ((PNSR * 0.5) +  (Color * 0.3) + (Black * 0.2) * coef
           
