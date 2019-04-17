/*****************************************************************************
 * Grupo: Luiz Gustavo Braganca dos Santos                                   *
 *        Maria Clara Dias                                                   *
 *        Pedro Augusto Prosdocimi                                           *
 * Materia: Computacao Paralela                                              *
 * Professor: Luis Fabricio Wanderley Goes                                   *
 * _________________________________________________________________________ *
 *                        MAQUINA LOCAL                                      *
 *                                                                           *
 *                      ---SEQUENCIAL---                                     *
 *                                                                           *
 * Rodando o comando: $ g++ main.cpp -o main                                 *
 *                    $ time ./main < ../dataset_iris                        *
 * real	0m5.731s                                                             *
 * user	0m0.730s                                                             *
 * sys	0m0.009s                                                             *
 * ------------------------------------------------------------------------- * 
 *                      ---PARALELO---                                       *
 *                                                                           *
 * Rodando o comando: $ g++ main.cpp -o main -fopenmp                        *
 *                    $ time ./main < ../dataset_iris                        *
 *                                                                           *
 * real	m.s                                                                  *
 * user	m.s                                                                  *
 * sys	m.s                                                                  *
 * _________________________________________________________________________ *
 *                        ---SPEEDUP---                                      *
 *                                                                           *
 *                 (/) =                                                     *
 *                                                                           *
 * _________________________________________________________________________ *
 *                      SERVIDOR PARCODE                                     *
 *                                                                           *
 *                      ---SEQUENCIAL---                                     *
 *                                                                           *
 * Rodando o comando: $ g++ main.cpp -o main                                 *
 *                    $ time ./main < ../dataset_iris                        *
 *                                                                           *
 * real	0m1.312s                                                             *
 * user	0m1.305s                                                             *
 * sys	0m0.000s                                                             *
 * ------------------------------------------------------------------------- * 
 *                      ---PARALELO---                                       *
 *                                                                           *
 * Rodando o comando: $ g++ main.cpp -o main -fopenmp                        *
 *                    $ time ./main < ../dataset_iris                        *
 *                                                                           *
 * real	m.s                                                                  *
 * user	m.s                                                                  *
 * sys	m.s                                                                  *
 * _________________________________________________________________________ *
 *                        ---SPEEDUP---                                      *
 *                                                                           *
 *                 (/) =                                                     *
 *                                                                           *
 *****************************************************************************/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include "MLP.h"
#include "neuronio.h"
#include "funcao_linear.h"
#include "funcao_tan_hiperbolica.h"

int main(int argc, char *argv[])
{
	srand(time(NULL));

	FILE* arq;

	// verifica a quantidade de argumentos
	if(argc < 2)
		arq = stdin; // ler da entrada padrao
	else
		arq = fopen(argv[1], "r"); // ler do arquivo passado como parametro

	int num_entradas, num_camadas_escondidas, neuronios_saida;
	int qte_amostras, qte_teste, max_epocas;
	double erro_min, taxa_aprendizagem, taxa_reducao_aprendizado;
	char str_funcao_ativacao[255];
	std::vector<int> neuronios_camadas_escondidas;
	
	// as linhas abaixo obtem as entradas de acordo com arquivo "config_instrucoes"
	
	fscanf(arq, "%d %d %d", &num_entradas, &num_camadas_escondidas, &neuronios_saida);
	
	neuronios_camadas_escondidas.push_back(num_entradas);
	
	// obtem a quantidade de neuronios de cada camada escondida
	for(int i = 0; i < num_camadas_escondidas; i++)
	{
		int n_neuronios;
		
		fscanf(arq, "%d", &n_neuronios);
		neuronios_camadas_escondidas.push_back(n_neuronios);
	}
	
	fscanf(arq, "%s", str_funcao_ativacao);
	fscanf(arq, "%d %d", &qte_amostras, &qte_teste);
	fscanf(arq, "%d %lf %lf %lf", &max_epocas, &erro_min, &taxa_aprendizagem, &taxa_reducao_aprendizado);
	FuncaoAtivacao * funcao_ativacao = NULL;

	// verifica a funcao de ativacao
	if(strcmp(str_funcao_ativacao, "hiperbolica") == 0)
		funcao_ativacao = new FuncaoTanHiperbolica(2);
	else if(strcmp(str_funcao_ativacao, "linear") == 0)
		funcao_ativacao = new FuncaoLinear();

	// cria uma instancia de MLP passando os parametros
	MLP mlp(qte_amostras, qte_teste, num_entradas, num_camadas_escondidas, 
				neuronios_saida, taxa_aprendizagem, taxa_reducao_aprendizado, 
				max_epocas, erro_min, funcao_ativacao, neuronios_camadas_escondidas);

	// vetores de entradas e saidas
	std::vector<std::vector<double> > amostras, saidas;

	// ler todas as entradas com suas respectivas saidas
	for(int i = 0; i < qte_amostras + qte_teste; i++)
	{
		std::vector<double> amostra_entradas, amostra_saidas;
		double valor;

		// ler uma entrada
		for(int j = 0; j < num_entradas; j++)
		{
			fscanf(arq, "%lf", &valor);
			amostra_entradas.push_back(valor);
		}

		amostras.push_back(amostra_entradas);

		// ler a saida
		for(int j = 0; j < neuronios_saida; j++)
		{
			fscanf(arq, "%lf", &valor);
			amostra_saidas.push_back(valor);
		}

		saidas.push_back(amostra_saidas);
	}
	
	// vetor de indices para embaralhar as entradas
	std::vector<int> vet_indices;
	for(int i = 0; i < qte_amostras + qte_teste; i++)
		vet_indices.push_back(i);
	
	// embaralha as entradas
	std::vector<std::vector<double> > amostras_random, saidas_random;
	for(int i = 0; i < qte_amostras + qte_teste; i++)
	{
		std::random_shuffle(vet_indices.begin(), vet_indices.end());
		amostras_random.push_back(amostras[vet_indices[i]]);
		saidas_random.push_back(saidas[vet_indices[i]]);
	}
	
	// treina a rede
	mlp.treinar(amostras_random, saidas_random);

	delete funcao_ativacao;

	if(arq != stdin)
		fclose(arq);

	return 0;
}
