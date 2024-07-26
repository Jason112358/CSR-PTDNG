import os
import umap
import matplotlib.pyplot as plt
import pandas as pd
# import umap.plot as uplt
import numpy as np


def embeddings2pic(embedding, name_list, name2label, config, model_name):
    plt.clf()
    # 绘制UMAP降维后的散点图
    umap_model = umap.UMAP(
        n_neighbors=config["umap"]["n_neighbors"],
        n_components=config["umap"]["n_components"],
        metric=config["umap"]["metric"],
        random_state=config["umap"]["random_state"],
    )
    embedded_data = umap_model.fit_transform(embedding)
    label2name = {value: key for key, value in name2label.items()}
    color_name_list = [label2name[label] for label in range(0, len(name2label))]
    plt.scatter(
        embedded_data[:, 0],
        embedded_data[:, 1],
        c=[name2label[name] for name in name_list],
        cmap=config["plt"]["cmap"],
        # s=config["plt"]["s"],
    )
    fc = plt.colorbar(boundaries=np.arange(0, len(color_name_list) + 1)-0.5) if config["plt"]["colorbar"] else None
    fc.set_ticks(range(0, len(color_name_list)))
    fc.set_ticklabels(color_name_list)
    plt.title(model_name + ' - ' + config["plt"]["title"])
    
    if config["show"]:
        plt.show()
    
    if config["save"]:
        os.makedirs(config["save_path"], exist_ok=True)
        plt.savefig(config["save_path"] + '/' + model_name + ".png", dpi=config["dpi"])

    plt.close()
