# graphics.R
# ==========
# Gera boxplots, matrizes de confusão e estatísticas (ANOVA/Scott-Knott)
# a partir dos resultados da validação cruzada.
#
# Chamado automaticamente por rodaCruzada.sh ao fim do experimento.
# Para rodar manualmente:
#   cd src && Rscript graphics.R

library(ggplot2)
library(gridExtra)
library(plyr)
library(dplyr)
library(stringr)
library(forcats)
library(scales)
library(reshape2)
library(data.table)
library(kableExtra)
library(ExpDes)
library(ExpDes.pt)

# ─────────────────────────────────────────────────────────────────
# MAPEAMENTO DE NOMES (nome interno → apelido para os gráficos)
# Adicione novas arquiteturas/otimizadores aqui se precisar.
# ─────────────────────────────────────────────────────────────────

arch_map <- list(
    alexnet                              = "AlexNet",
    coat_tiny                            = "CoaT",
    convnext_base                        = "ConvNeXt",
    densenet201                          = "DenseNet201",
    lambda_resnet26rpt_256               = "LambdaResNet",
    lamhalobotnet50ts_256                = "LamHaloBotNet",
    maxvit_rmlp_tiny_rw_256              = "MaxViT",
    mobilenetv3                          = "MobileNetV3",
    resnet18                             = "ResNet18",
    resnet50                             = "ResNet50",
    resnet101                            = "ResNet101",
    sebotnet33ts_256                     = "SEBotNet",
    swinv2_base_window16_256             = "SwinV2-W16",
    swinv2_cr_base_224                   = "SwinV2-CR",
    vgg19                                = "VGG19",
    vit_relpos_base_patch32_plus_rpn_256 = "ViTRelPosRPN",
    ielt                                 = "IELT",
    default_siamese                      = "Siamese"
)

optim_map <- list(
    sgd     = "SGD",
    adam    = "Adam",
    adamw   = "AdamW",
    adagrad = "Adagrad",
    lion    = "Lion",
    sam     = "SAM"
)

# ─────────────────────────────────────────────────────────────────
# LEITURA DOS DADOS
# ─────────────────────────────────────────────────────────────────

results_path <- "../results_dl/results.csv"
if (!file.exists(results_path)) {
    stop(paste("Arquivo não encontrado:", results_path))
}

data <- read.table(results_path, sep=",", header=TRUE)
data$original_arch  <- data$architecture
data$original_optim <- data$optimizer

# Aplica os apelidos
for (key in names(arch_map)) {
    data[data$architecture == key, "architecture"] <- arch_map[[key]]
}
for (key in names(optim_map)) {
    data[data$optimizer == key, "optimizer"] <- optim_map[[key]]
}

cat(sprintf("Resultados carregados: %d linhas\n", nrow(data)))

# ─────────────────────────────────────────────────────────────────
# BOXPLOTS
# ─────────────────────────────────────────────────────────────────

metrics <- c("precision", "recall", "fscore")

y_min <- min(data[, metrics]) - 0.01
y_max <- max(data[, metrics]) + 0.01

for (lr in unique(data$learning_rate)) {
    df_lr  <- data[data$learning_rate == lr, ]
    plots  <- list()

    for (i in seq_along(metrics)) {
        metric <- metrics[i]
        title  <- sprintf("Arq × Opt  |  lr = %s  |  %s",
                          format(lr, scientific=TRUE), metric)

        plots[[i]] <- ggplot(df_lr, aes_string(x="architecture", y=metric, fill="optimizer")) +
            geom_boxplot(outlier.shape=21) +
            ylim(y_min, y_max) +
            scale_fill_brewer(palette="Set2") +
            labs(title=title, x="Arquitetura", y=metric, fill="Otimizador") +
            theme_minimal() +
            theme(
                plot.title    = element_text(hjust=0.5, size=10),
                axis.text.x   = element_text(angle=30, hjust=1),
                legend.position = "right"
            )
    }

    lr_tag  <- sub("0\\.", "_", sprintf("%f", lr))
    outfile <- paste0("../results_dl/boxplot", lr_tag, ".png")
    g       <- grid.arrange(grobs=plots, ncol=1)
    ggsave(outfile, g, width=12, height=10)
    cat(sprintf("Boxplot salvo: %s\n", outfile))
}

# ─────────────────────────────────────────────────────────────────
# ESTATÍSTICAS DESCRITIVAS
# ─────────────────────────────────────────────────────────────────

options(width=10000)
dt <- data.table(data)

make_stats <- function(metric) {
    dt[, .(
        median = median(get(metric)),
        IQR    = IQR(get(metric)),
        mean   = mean(get(metric)),
        sd     = sd(get(metric))
    ), by = .(learning_rate, architecture, optimizer)]
}

prec_stats <- make_stats("precision")
rec_stats  <- make_stats("recall")
f1_stats   <- make_stats("fscore")

sink("../results_dl/statistics.txt")
cat("\n[ Precision ]─────────────────────────────────\n"); print(prec_stats)
cat("\n[ Recall ]────────────────────────────────────\n"); print(rec_stats)
cat("\n[ F-score ]───────────────────────────────────\n"); print(f1_stats)
sink()

sink("../results_dl/statistics_for_latex.txt")
cat(kbl(prec_stats, caption="Precision", format="latex",
        col.names=c("LR","Arquitetura","Otimizador","Mediana","IQR","Média","DP"), align="r"))
cat(kbl(rec_stats,  caption="Recall",    format="latex",
        col.names=c("LR","Arquitetura","Otimizador","Mediana","IQR","Média","DP"), align="r"))
cat(kbl(f1_stats,   caption="F-score",   format="latex",
        col.names=c("LR","Arquitetura","Otimizador","Mediana","IQR","Média","DP"), align="r"))
sink()
cat("Estatísticas salvas.\n")

# ─────────────────────────────────────────────────────────────────
# MATRIZES DE CONFUSÃO — melhor configuração por métrica
# ─────────────────────────────────────────────────────────────────

median_vals <- dt[, .(
    precision = median(precision),
    recall    = median(recall),
    fscore    = median(fscore)
), by = .(learning_rate, architecture, optimizer, original_arch, original_optim)]

classes <- as.vector(system("ls -1 ../data/all", intern=TRUE))
folds   <- sprintf("fold_%d", seq_len(max(data$run)))

for (metric in metrics) {
    out_dir <- paste0("../results_dl/matrices_for_best_", metric)
    dir.create(out_dir, showWarnings=FALSE)

    best <- median_vals %>% filter(.data[[metric]] == max(.data[[metric]]))
    if (nrow(best) == 0) next
    best <- best[1, ]   # desempate: pega o primeiro

    cat(sprintf("Melhor %s → arch=%s opt=%s lr=%s\n",
                metric, best$architecture, best$optimizer, best$learning_rate))

    csv_name <- sprintf("%s_%s_%s_MATRIX.csv",
                        best$original_arch,
                        best$original_optim,
                        format(best$learning_rate, scientific=FALSE))

    mean_matrix <- NULL

    for (fold in folds) {
        fold_num <- sub("fold_", "", fold)
        csv_path <- file.path("../resultsNfolds", fold, "matrix",
                              paste0(fold_num, "_", csv_name))

        if (!file.exists(csv_path)) next

        mat        <- read.table(csv_path, sep=",", header=FALSE)
        filtered   <- mat[-1, -1]
        normalized <- filtered / sum(filtered)

        mean_matrix <- if (is.null(mean_matrix)) normalized else mean_matrix + normalized

        rounded     <- round(normalized, 2)
        colnames(rounded) <- classes
        cm_long     <- reshape2::melt(cbind(classes=classes, rounded))
        cm_long     <- cm_long %>% mutate(
            variable = factor(variable),
            classes  = factor(classes, levels=rev(unique(classes)))
        )

        title_fold <- sprintf("Dobra %s: %s, %s, LR=%s",
                              fold_num, best$architecture, best$optimizer,
                              format(best$learning_rate, scientific=TRUE))

        g <- ggplot(cm_long, aes(x=variable, y=classes, fill=value)) +
            geom_tile() + geom_text(aes(label=value)) +
            xlab("Predito") + ylab("Real") + ggtitle(title_fold) +
            labs(fill="Proporção") +
            theme_minimal() +
            theme(axis.text.x=element_text(angle=45, hjust=1), aspect.ratio=1)

        ggsave(
            file.path(out_dir, sprintf("%s_%s_%s_%s_cm.png",
                                       best$architecture, best$optimizer,
                                       best$learning_rate, fold)),
            g, width=6, height=5
        )
    }

    # Matriz média entre dobras
    if (!is.null(mean_matrix)) {
        mean_matrix <- mean_matrix / length(folds)
        rounded     <- round(mean_matrix, 2)
        colnames(rounded) <- classes
        cm_long     <- reshape2::melt(cbind(classes=classes, rounded))
        cm_long     <- cm_long %>% mutate(
            variable = factor(variable),
            classes  = factor(classes, levels=rev(unique(classes)))
        )
        title_mean <- sprintf("Média: %s, %s, LR=%s",
                               best$architecture, best$optimizer, best$learning_rate)
        g <- ggplot(cm_long, aes(x=variable, y=classes, fill=value)) +
            geom_tile() + geom_text(aes(label=value)) +
            xlab("Predito") + ylab("Real") + ggtitle(title_mean) +
            labs(fill="Proporção") +
            theme_minimal() +
            theme(axis.text.x=element_text(angle=45, hjust=1))
        ggsave(
            file.path(out_dir, sprintf("%s_%s_%s_MEAN_cm.png",
                                       best$architecture, best$optimizer, best$learning_rate)),
            g, width=6, height=5
        )
        cat(sprintf("  Matrizes de confusão salvas em %s/\n", out_dir))
    }
}

# ─────────────────────────────────────────────────────────────────
# ANOVA E TESTES DE COMPARAÇÃO MÚLTIPLA
# Aplica 1-way, 2-way ou 3-way dependendo de quantos fatores
# têm mais de um valor distinto nos dados.
# ─────────────────────────────────────────────────────────────────

possible_factors <- c("architecture", "optimizer", "learning_rate")
active_factors   <- possible_factors[sapply(possible_factors, function(f) length(unique(data[[f]])) > 1)]

cat(sprintf("\nFatores ativos para ANOVA: %s\n", paste(active_factors, collapse=", ")))

if (length(active_factors) == 1) {

    sink("../results_dl/anova.txt")
    f <- active_factors[1]
    for (metric in metrics) {
        cat(sprintf("\n──── %s | %s ────\n", toupper(metric), f))
        aov_res <- aov(as.formula(paste(metric, "~", f)), data=data)
        print(summary(aov_res))
        cat("\nTukey HSD:\n")
        print(TukeyHSD(aov_res))
    }
    sink()

} else if (length(active_factors) == 2) {

    sink("../results_dl/anova.txt")
    for (metric in metrics) {
        cat(sprintf("\n──── %s ────\n", toupper(metric)))
        fat2.dic(
            data[[active_factors[1]]],
            data[[active_factors[2]]],
            data[[metric]],
            quali=c(TRUE, TRUE), mcomp="sk"
        )
    }
    sink()

} else if (length(active_factors) == 3) {

    sink("../results_dl/anova.txt")
    for (metric in metrics) {
        cat(sprintf("\n──── %s ────\n", toupper(metric)))
        fat3.dic(
            data[[active_factors[1]]],
            data[[active_factors[2]]],
            data[[active_factors[3]]],
            data[[metric]],
            quali=c(TRUE, TRUE, TRUE), mcomp="sk"
        )
    }
    sink()

} else {
    cat("Número de fatores inválido para ANOVA.\n")
}

cat("\ngraphics.R concluído.\n")
