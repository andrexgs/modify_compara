# Script para contar as imagens nas pastas dentro data
# Usado apenas detectar algum erro.

cd ../data/


# Cria caso não exista
if [[! -d "./test"]]; then
  mkdir -p test
fi

echo 'Contagem Todas Juntas (ALL)'
for i in ./all/* 
do
  if [ -d "${i}" ] ; then
    echo $i " = "`ls "${i}" | wc -l`
  fi
done

echo 'Contagem Treinamento'
for i in ./train/* 
do
  if [ -d "${i}" ] ; then
    echo $i " = "`ls "${i}" | wc -l`
  fi
done

echo 'Contagem Teste'
for i in ./test/* 
do
  if [ -d "${i}" ] ; then
    echo $i " = "`ls "${i}" | wc -l`
  fi
done

echo 'Contagem Dobras'
for dobra in ./dobras/* 
do
  echo 'Na dobra ' $dobra
  for classe in ${dobra}/*
  do
    echo $classe " = "`ls "${classe}" | wc -l`
  done
done

echo 'Contagem Dobras Sinteticas'
for dobra in ./dobras_sinteticas/*
do
  echo 'Na dobra sintetica ' $dobra
  for classe in ${dobra}/*
  do
    echo $classe " = "`ls "${classe}" | wc -l`
  done
done

cd ../utils/
