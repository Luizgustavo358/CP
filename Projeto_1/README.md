## Sobre o K-Means

O KMeans é um algoritmo que é utilizado para calcular centroides de alguns clusters. Uma das aplicações dele é a verificação de zonas de interesse, por exemplo polos de furtos, que é adicionado um novo 'ponto' (incidente), a área onde ocorreu o crime (cluster). Junto a isso, o algoritmo realiza recalculos de distância entre cluters, aprimorando a fronteira e a torna mais precisa em relação à zona de interesse.

### Como Compilar:

- Sequencial:

```sh
$ gcc kmeans.cpp -o kmeans
```

- Paralelo:

```sh
$ gcc kmeans.cpp -o kmeans -fopenmp
```

### Como Rodar o Código:

```sh
$ time ./kmeans < datasets/pub.in
```
    
________________________________________________________________________________
Referência: https://github.com/marcoscastro/kmeans