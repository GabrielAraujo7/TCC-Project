def identificar_nicho(texto):
    texto = texto.lower()
    if any(p in texto for p in ["whey", "creatina", "academia", "suplemento"]):
        return "Fitness"
    elif any(p in texto for p in ["fralda", "bebê", "mamadeira", "lenço"]):
        return "Infantil"
    elif any(p in texto for p in ["caderno", "caneta", "mochila", "escolar"]):
        return "Escolar"
    else:
        return "Outro"
