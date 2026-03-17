# Ordena o arquivo ../results_dl/results.csv pelo valor da medida-F
# Vai gerar o arquivo ../results_dl/results_ordered.csv e também
# mostrar na tela os 10 primeiros do ranking

# Extrai o cabeçalho
echo "Dobra,LR,Otim,Arquit,F-Score" > ../results_dl/results_ordered.csv

# Processa o restante do arquivo e adiciona ao arquivo de saída
tail -n +2 ../results_dl/results.csv | 
cut -d ',' -f 1,2,3,4,7 | 
sort -t ',' -k5 -r >> ../results_dl/results_ordered.csv 

# Exibe o resultado na forma de tabela (10 primeiras linhas apenas)
head ../results_dl/results_ordered.csv | column -s, -t 
