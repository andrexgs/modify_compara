# install_R_packages.R
# ====================
# Instala todos os pacotes R necessários para o graphics.R.
#
# Uso:
#   sudo Rscript install_R_packages.R
#
# Nota: sudo é necessário para instalar na biblioteca do sistema.
# Para instalar apenas para o usuário atual, remova o sudo.

pkgs <- c(
    "ggplot2",      # gráficos
    "gridExtra",    # múltiplos gráficos na mesma figura
    "plyr",         # manipulação de dados
    "dplyr",        # manipulação de dados (versão moderna)
    "stringr",      # manipulação de strings
    "forcats",      # manipulação de fatores
    "scales",       # formatação de eixos
    "reshape2",     # pivot de dataframes
    "data.table",   # leitura e manipulação eficiente de tabelas
    "kableExtra",   # tabelas LaTeX / HTML
    "ExpDes",       # ANOVA e testes de comparação múltipla
    "ExpDes.pt"     # versão em português do ExpDes
)

# Instala apenas os pacotes que ainda não estão instalados
novos <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]

if (length(novos) > 0) {
    cat(sprintf("Instalando %d pacote(s): %s\n", length(novos), paste(novos, collapse=", ")))
    install.packages(novos, repos="https://cloud.r-project.org", dependencies=TRUE)
} else {
    cat("Todos os pacotes já estão instalados.\n")
}

# Verifica se todos foram instalados com sucesso
faltando <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if (length(faltando) > 0) {
    stop(sprintf("Falha ao instalar: %s", paste(faltando, collapse=", ")))
} else {
    cat("Instalação concluída com sucesso.\n")
}
