# Remove da pasta ../data/all/ todas as pastas que tem menos que
# um número mínimo de arquivos

if [ $# -eq 0 ]; then
    echo "Faltou passar o mínimo de imagens que deve ser usado"
    exit 1
fi

minimo=$1

echo "Removendo pastas com menos que $minimo arquivos"

# Remove as pastas com menos que o minimo de arquivos
# entre as subpastas dentro de ../data/all/
for pasta in $(ls ../data/all/); do
    n=$(ls ../data/all/$pasta | wc -l)
    echo "Pasta $pasta tem $n arquivos"
    if [ $n -lt $minimo ]; then
        echo "REMOVIDA !!!" 
        rm -r ../data/all/$pasta
    fi




done