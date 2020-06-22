from src.MLANE import MLANE
import networkx as nx


if __name__ == "__main__":
    G=nx.read_edgelist('../data/citeseer/citeseer.edgelist',
                         create_using = nx.Graph(), nodetype = None, data = [('weight', int)])
    graph_label_path = '../data/citeseer/citeseer_labels.txt'

    '''
    initialize model
    '''
    model = MLANE(G, graph_label_path, walk_length=40, num_walks=10, window_size=10, embed_size=512, train_percent=0.8)

    '''
    train policy
    '''
    model.policy_train(i_episode=5, epoch=50)
    model.save_model('citeseer')

    '''
    learn embeddings
    '''
    model.load_model('citeseer')
    model.train()
    model.save_embeddings('citeseer')

    '''
    evaluate embeddings
    '''
    embeddings = model.load_embeddings('citeseer')
    model.evaluate_embeddings(embeddings)