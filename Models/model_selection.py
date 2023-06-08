    # Сканирует папку с весами и удаляет всё, что хуже:
    # - max_best лучшие веса модели
    # - max_n_weights новые веса, которые будут сравниваться с лучшими
    def _remove_old_weights(self, max_n_weights, max_best=5):
        w_list = {}
        w_list['all'] = [w for w in self.callback_paths['weights'].iterdir() if '.pth' in w.name]
        w_list['best'] = [w for w in w_list['all'] if 'best' in w.name]
        w_list['others'] = [w for w in w_list['all'] if w not in w_list['best']]
        # удаление весов, которые хуже
        epochs_set = {}
        epochs_set['best'] = list(
            set([self.epoch_n_from_weights_name(w.name) for w in w_list['best']])
        )
        epochs_set['others'] = list(
            set([self.epoch_n_from_weights_name(w.name) for w in w_list['others']])
        )
        keep_max = {'best': max_best, 'others': max_n_weights}
        for type in ['others', 'best']:
            if len(epochs_set[type]) > keep_max[type]:
                epoch_list = np.sort(epochs_set[type])[::-1]
                epoch_list = epoch_list[0: keep_max[type]]
                for w in w_list[type]:
                    if self.epoch_n_from_weights_name(w.name) not in epoch_list:
                        w.unlink()

print("The best model is SubPixelCNN")
