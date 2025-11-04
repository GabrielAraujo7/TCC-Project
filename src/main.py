from nlp import identificar_nicho
from map import gerar_mapa
from clustering_pipeline import gerar_regioes_ideais

def processar_requisicao(produto, filtros):
    nicho = identificar_nicho(produto)
    regioes = gerar_regioes_ideais(produto, filtros)
    mapa = gerar_mapa(regioes)  
    return nicho, mapa