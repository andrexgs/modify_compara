#
#    Script - split data between train and test
#
#    Name: splitFold.sh
#    Author: Hemerson Pistori (pistori@ucdb.br)
#    Author: Maxwell Sampaio (maxs.santos@gmail.com)
#

############################################################
# Help                                                     #
############################################################
Help() {
   # Display Help
   #echo "A description of the script functions here:"
   echo "Syntax: ./splitFolds.sh [-s|S|h|H] [-k k_folds]"
   echo "options:"
   echo "h|H            Print this Help."
   echo "s|S            Dataset composed of subsets of items."
   echo "k k_folds      Set the number of folds."
}

############################################################
# Main program                                             #
############################################################

echo "[SCRIPT SPLIT DATA IN N FOLDS] Initializing..."

dir_all="../data/all"
dir_train="../data/train"
dir_dobras="../data/dobras/"
subset=0

# IMPORTANTE: 3 dobras é muito pouco. Usei apenas para rodar mais
# rapidamente um exemplo.
ndobras=3

# Get the flags
while getopts "k:s|S|h|H" flag; 
do
   case "${flag}" in
      k) ndobras=${OPTARG} ;;
      [sS])
         subset=1 ;;
      [hH]) # display Help
         Help
         exit 0
         ;;
      *)
         echo "[SCRIPT SPLIT DATA IN N FOLDS] Error: Invalid option. (${flag})"
         Help
         exit 1
         ;;
   esac
done

folds=()
for ((i=1; i<=$ndobras; i+=1)); do folds+=("fold_${i}"); done

mkdir -p $dir_train
rm -rf $dir_train/*

cp -R $dir_all/* $dir_train

mkdir -p $dir_dobras/
rm -rf $dir_dobras/*

#realiza o agrupamento das imagens de mesmo individuo criando
#uma pasta em cada uma das classes para cada um dos individuos
#e movendo suas respectivas imagens para esta pasta
if [[ $subset -eq 1 ]]; then
   for dir_class in $(ls $dir_train); 
   do
      ids=()
      for img in $(ls ${dir_train}"/"${dir_class});
      do
         string=$img"_"
         
         #Split the text based on the delimiter
         myarray=()
         while [[ $string ]]; do
            myarray+=( "${string%%"_"*}" )
            string=${string#*"_"}
         done

         if [[ ! ${ids[*]} =~ (^|[[:space:]])"${myarray[-2]}"($|[[:space:]]) ]]; then
            ids[${#ids[*]}]=${myarray[-2]}

            mkdir  -p  $dir_train/${dir_class}/${myarray[-2]}
         fi
         mv  $dir_train/$dir_class/$img  $dir_train/$dir_class/${myarray[-2]}/$img         
      done
   done
fi

for dir_class in $(ls $dir_train); do
   echo "[SCRIPT SPLIT DATA] Spliting class -" $dir_class
   
   total_itens=$(ls $dir_train/$dir_class | wc -l)
   total_por_dobra_float=$(echo "scale=2; ($total_itens/$ndobras)" | bc -l)
   total_por_dobra=${total_por_dobra_float%.*}

   echo 'Itens por dobra nesta classe: ' $total_por_dobra

   for fold in "${folds[@]}"; do
      echo "[SCRIPT SPLIT DATA] Creating " $fold
      dir_fold=${dir_dobras}${fold}
      mkdir -p $dir_fold
      
      mkdir -p $dir_fold/$dir_class

      counter=0
      arrayFiles=$(ls $dir_train/$dir_class | sort -R)
      for file in $arrayFiles; do
         let "counter += 1"
         if [[ $counter -le $total_por_dobra ]]; then
            # echo 'Count = ' $counter 'Moving ' $dir_train/$dir_class/$file ' to ' $dir_fold/$dir_class/$file
            mv $dir_train/$dir_class/$file $dir_fold/$dir_class/$file
         fi
      done
   done

   # Move o que sobrou por conta do arredondamento para a última dobra
   mv $dir_train/$dir_class/* $dir_fold/$dir_class/ 2> /tmp/splitFold.log

done

#reorganiza o conjunto de dados do diretório dir_dobras 
#para ter apenas imagens e não subdiretórios dentro de cada uma das classes
if [[ $subset -eq 1 ]]; then
   for dir_fold in $(ls $dir_dobras); 
   do
      for dir_class in $(ls ${dir_dobras}${dir_fold});
      do 
         for dir_item in $(ls ${dir_dobras}${dir_fold}"/"$dir_class);
         do 
            #caso realmente existam diretórios de agrupamento das imagens 
            #de determinado item dentro das classes de cada dobra
            if [ -d ${dir_dobras}${dir_fold}"/"$dir_class"/"$dir_item ]; then
               mv ${dir_dobras}${dir_fold}/${dir_class}/${dir_item}/* ${dir_dobras}${dir_fold}/${dir_class}/.
               rm -rf ${dir_dobras}${dir_fold}/$dir_class/$dir_item
            fi
         done
      done
   done
fi

echo "[SCRIPT SPLIT DATA] OK! DONE."