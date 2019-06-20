class Config(object):
    ds_path = 'npydata'
    doc_totalword_file = 'npydata/docs_len.npy'
    logging_file = False
    batch_size = 64
    loc_std = 0.22

    window_size_exw = 20
    extract_words_num = window_size_exw * 2
    random_ws_exw = 10
    num_word = 10000
    word_vector = 100

    filter_nums = 128
    hl_size = 128
    g_size = 256
    cell_output_size = 256
    loc_dim = 1

    cell_size = 256
    cell_out_size = cell_size

    num_glimpses = 6
    num_classes = 11
    max_grad_norm = 5.
    kernel_size1 = 3
    kernel_size2 = 4
    kernel_size3 = 5
    maxpool_size1 = extract_words_num - kernel_size1 + 1
    maxpool_size2 = extract_words_num - kernel_size2 + 1
    maxpool_size3 = extract_words_num - kernel_size3 + 1
    #add

    step = 10001
    val_freq = 200
    lr_start = 1e-3
    lr_min = 1e-4

    # Monte Carlo sampling
    M = 10
