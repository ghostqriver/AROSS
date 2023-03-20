def gen_file_name(args):
    '''
    Generate a string by the given parameters of COS
    '''
    fn = ''
    # exclude 'all_safe_gen','half_safe_gen' in automatical way, or it will error because the generator function name is invalid when creating a file
    for str_ in [j[0]+str(j[1]) for j in list(filter(lambda i:i[0] not in ['all_safe_gen','half_safe_gen'],[(i,args['args'][i]) for i in args['args']]))]:
        fn = fn  + str_ +'_'
    # add generator in
    if 'all_safe_gen' in args['args'].keys():
        if args['args']['all_safe_gen'] == G.Gaussian_Generator:
            fn = fn + 'all_safe_genGaussian' + '_'
    else:
        fn = fn + 'all_safe_genSMOTE' + '_'
            
    if 'half_safe_gen' in args['args'].keys():
        if args['args']['half_safe_gen'] == G.Gaussian_Generator:
            fn = fn + 'half_safe_genGaussian' + '_'
    else:
        fn = fn + 'half_safe_genSMOTE' + '_'
    return fn

def get_cos_para(args):
    
    c = args['c']
    N = args['N']
    alpha = args['alpha']

    if 'linkage' in args.keys():
        linkage = args['linkage']
    else:
        linkage = 'single'

    if 'l' in args.keys():
        L = args['l']
    else:
        L = 2

    if 'shrink_half' in args.keys():
        shrink_half = args['shrink_half']
    else:
        shrink_half = None

    if 'expand_half' in args.keys():
        expand_half = args['expand_half']
    else:
        expand_half = None

    if 'all_safe_weight' in args.keys():
        all_safe_weight = args['all_safe_weight']
    else:
        all_safe_weight = 2

    if 'all_safe_gen' in args.keys():
        all_safe_gen = args['all_safe_gen']
    else:
        all_safe_gen = G.Smote_Generator
        
    if 'half_safe_gen' in args.keys():
        half_safe_gen = args['half_safe_gen']
    else:
        half_safe_gen = G.Smote_Generator   

    if 'gaussian_scale' in args.keys():
        Gaussian_scale = args['gaussian_scale']
    else:
        Gaussian_scale = 0.8  

    if 'ir' in args.keys():
        IR = args['ir']
    else:
        IR=1

    if 'minlabel' in args.keys():
        minlabel = args['minlabel']
    else:
        minlabel = None

    if 'majlabel' in args.keys():
        majlabel = args['majlabel']
    else:
        majlabel = None

    if 'visualize' in args.keys():
        visualize = args['visualize']
    else:
        visualize = False

    return N,c,alpha,linkage,L,shrink_half,expand_half,all_safe_weight,all_safe_gen,half_safe_gen,Gaussian_scale,IR,minlabel,majlabel,visualize

def show_baseline_cos(dataset,random_state=None,pos_label=None,**args): 
    
    model = 'cos'

    args['args']['visualize'] = True

    X,y = read_data(dataset)

    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=random_state)

    pos_label = get_labels(y_test)[0]

    X_oversampled,y_oversampled,_,_ = oversampling(model,X_train,y_train,args['args'])

    plt.show()
    

