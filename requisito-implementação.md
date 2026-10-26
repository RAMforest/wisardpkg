# WiSARD por épocas — requisitos de implementação

Oct 17, 2026 · @Yasmin Romeiro

## 1. Escopo e objetivo

Implementar, sobre a WiSARD clássica, um modo de treinamento por épocas e mini-batches que corrige contadores já gravados sem gradiente, mantendo intactas a estrutura (RAMs, discriminadores, endereçamento) e a inferência (votação com bleaching). O resultado é uma WiSARD comum, legível e compatível com as ferramentas existentes; só a política de escrita muda.

**Base de código:** a implementação é feita sobre o [wisardpkg](https://github.com/IAZero/wisardpkg), como extensão de suas classes `Wisard`, `Discriminator` e `RAM`, e não como pacote novo. Toda a estrutura de dados, o endereçamento, o bleaching iterativo e a serialização existentes são reaproveitados.

**Dentro do escopo**

- Classificar cada exemplo antes de gravá-lo e penalizar, multiplicativamente, os endereços ativos do discriminador que venceu indevidamente ou do segundo colocado quando a margem é baixa.
- Divisão treino / validação / teste, com a validação escolhendo hiperparâmetros e o teste usado uma vez.
- Métricas por época para oscilação e esquecimento.

**Fora do escopo**

- SWiSARD, RAMs de regra e ajuste de α (linha separada).
- Contadores negativos ou evidência negativa.
- Mental images (DRASiW) durante o treino, por custo.
- Qualquer uso de gradiente, backpropagation ou straight-through estimator.

## 2. Notação e definições

| Símbolo | Significado |
| --- | --- |
| x, y | exemplo binarizado e sua classe |
| D | conjunto de discriminadores (um por classe) |
| n | número de RAMs por discriminador |
| a\_r(x) | endereço da RAM r ativado por x |
| m\_{d,r}\[a\] | contador (inteiro ≥ 0) da RAM r do discriminador d no endereço a |
| N\_d | número de exemplos da classe d já vistos |
| b | limiar de bleaching fixo usado durante o treino |
| R\_d(x) | resposta do discriminador d: número de RAMs com m ≥ b no endereço ativo |
| w, l | vencedor e segundo colocado em R\_d(x) |
| C | confiança da decisão |
| ρ | dureza da penalização, ρ ∈ (0, 1) |
| τ | margem mínima de confiança exigida, τ ∈ \[0, 1) |
| B\_1 … B\_k | mini-batches de uma época |
| E | número máximo de épocas |

Resposta e confiança:

```latex
R_d(x) = \sum_{r=1}^{n} \mathbb{1}\big[\, m_{d,r}[a_r(x)] \ge b \,\big], \qquad C = \frac{R_w - R_l}{R_w}
```

Definições:

- **Erro**: w ≠ y.
- **Acerto com margem baixa**: w = y e C < τ.
- **Penalizar um endereço**: multiplicar seu contador por um fator em (0, 1); nunca produz valor negativo.
- **Reforçar**: incrementar o contador em 1 (treino WiSARD normal).

## 3. Requisitos funcionais

**Dados**

- **RF-01** O sistema recebe três partições disjuntas: treino, validação e teste. A validação nunca é gravada nas RAMs; o teste só é avaliado após o fim do treino.
- **RF-02** A cada época, o treino é embaralhado (semente fixa) e particionado em mini-batches de tamanho configurável.

**Classificação antes da gravação**

- **RF-03** Para cada exemplo (x, y) do batch, o sistema calcula R\_d(x) para todo d, em um limiar b fixo, antes de qualquer escrita. O bleaching iterativo (subir b até desempatar) não é usado no treino.
- **RF-04** O sistema identifica w (vencedor), l (segundo colocado) e C. Empate em R: w é o discriminador de menor índice; C = 0.

**Atualização**

- **RF-05** Em todo exemplo, o sistema reforça os endereços ativos do discriminador correto y em +1 (treino normal).
- **RF-06** Se w ≠ y (erro), o sistema penaliza cada endereço ativo de w com o fator (1 − ρ\_ef), onde ρ\_ef é a dureza efetiva (RF-08).
- **RF-07** Se w = y e C < τ, o sistema penaliza endereços ativos de l com fator (1 − ρ\_ef · (τ − C)/τ). Nenhuma atualização além do reforço ocorre quando C ≥ τ.
- **RF-08** Dureza efetiva: ρ\_ef = ρ\_t para erro; no erro, opcionalmente modulada por (R\_w − R\_y)/R\_w. ρ\_t decai por época conforme RF-11.
- **RF-09** Seleção de endereços a penalizar, configurável: (a) todos os ativos; (b) os κ de menor contador; (c) os κ com menor razão m/b. Padrão: (a) para erro, (c) para margem baixa.
- **RF-10** Contadores nunca ficam negativos. Durante o treino os contadores são float; ao final do treino (e na serialização) são arredondados para inteiro, e a inferência usa inteiros. Inteiro com arredondamento estocástico durante o treino fica como opção.

**Épocas e parada**

- **RF-11** Esquema de decaimento de ρ configurável: constante, ρ\_0/(1 + t) ou ρ\_0 · γ^t.
- **RF-12** Ao fim de cada época, o sistema avalia a validação e registra as métricas da seção 7.
- **RF-13** Parada por E máximo ou por early stopping (sem melhora na validação por p épocas); o modelo entregue é o da melhor época.

**Inferência e persistência**

- **RF-14** O modelo treinado é classificado pelo caminho padrão da WiSARD (incluindo bleaching iterativo), sem código novo na inferência.

## 4. Requisitos não funcionais

- **RNF-01 Custo por exemplo** O(n): uma classificação (leitura de n endereços por discriminador) e no máximo duas escritas de n endereços. Nenhuma estrutura cresce com N ou com 2^(bits por RAM) além das RAMs já existentes.
- **RNF-02 Custo por época** O(|treino| · n · |D|) para o treino mais uma passada de classificação na validação. Nenhuma passada adicional para métricas.
- **RNF-03 Memória** Igual à WiSARD one-shot, mais um snapshot do melhor modelo para early stopping.
- **RNF-04 Compatibilidade** O modo é opcional e desligado por padrão; com E = 1 e ρ = 0 o resultado é idêntico ao treino one-shot.
- **RNF-05 Reprodutibilidade** Toda aleatoriedade (embaralhamento, arredondamento estocástico, desempates) deriva de uma única semente; mesma semente e mesmos dados produzem contadores idênticos.
- **RNF-06 Observabilidade** As métricas da seção 7 são gravadas por época em formato tabular (CSV ou JSON) e podem ser desligadas.
- **RNF-07 Isolamento** A lógica do código de inferência existente não é alterado. No entanto, é preciso ajustar para trabalhar com valores float salvos nas RAMS.
- **RNF-08 Base de código** O desenvolvimento ocorre no wisardpkg: o laço de épocas e a penalização são implementados no núcleo C++ (mesma camada de `train` e `classify`) e expostos em Python pelos bindings existentes, sem dependências novas. A RAM continua um mapa esparso endereço → contador; o contador passa a aceitar float durante o treino (RF-10) sem alterar o formato serializado, que permanece inteiro.
- **RNF-09 Entrega** A funcionalidade entra como branch do repositório, com os testes da seção 10 na suíte existente e um exemplo em Python no diretório de exemplos.

## 5. Algoritmo

O laço externo percorre épocas; o interno classifica, penaliza e grava cada exemplo. A penalização sempre precede o reforço, para que o próprio exemplo não infle R\_y antes de medir a margem.

```python
def train_epochs(model, train, val, cfg, rng):
    best, best_score, since_best = snapshot(model), -inf, 0
    for t in range(cfg.E):
        rho_t = cfg.decay(cfg.rho0, t)
        for batch in minibatches(shuffle(train, rng), cfg.batch_size):
            for x, y in batch:
                R = model.responses(x, b=cfg.b)            # R[d] para todo d, sem bleaching iterativo
                w, l = top2(R)                              # desempate: menor índice
                C = (R[w] - R[l]) / R[w] if R[w] > 0 else 0.0

                if w != y:                                  # erro
                    rho_ef = rho_t * ((R[w] - R[y]) / R[w] if cfg.modulate else 1.0)
                    addrs = select(model, w, x, cfg.select_error)
                    penalize(model, w, addrs, factor=1 - rho_ef, rng)
                elif C < cfg.tau:                           # acerto com margem baixa
                    rho_ef = rho_t * (cfg.tau - C) / cfg.tau
                    addrs = select(model, l, x, cfg.select_margin)
                    penalize(model, l, addrs, factor=1 - rho_ef, rng)

                model.train_one(x, y)                       # reforço: m[y, r, a_r(x)] += 1

        score, metrics = evaluate(model, val, cfg.b)
        log(t, metrics)
        if score > best_score:
            best, best_score, since_best = snapshot(model), score, 0
        else:
            since_best += 1
            if since_best >= cfg.patience:
                break
    return best


def penalize(model, d, addrs, factor, rng):
    for r, a in addrs:
        m = model.ram[d][r][a]
        new = m * factor
        if cfg.counter_mode == "float":
            model.ram[d][r][a] = new
        else:                                               # inteiro com arredondamento estocástico
            lo = floor(new)
            model.ram[d][r][a] = lo + (1 if rng.random() < new - lo else 0)
        if model.ram[d][r][a] == 0:
            del model.ram[d][r][a]                          # mantém a RAM esparsa


def select(model, d, x, mode):
    active = [(r, a_r(x)) for r in range(n) if a_r(x) in model.ram[d][r]]
    if mode == "all":
        return active
    key = (lambda ra: model.ram[d][ra[0]][ra[1]]) if mode == "lowest_count" \
          else (lambda ra: model.ram[d][ra[0]][ra[1]] / cfg.b)
    return sorted(active, key=key)[:cfg.kappa]
```

Invariantes a garantir em código:

- Nenhum contador é escrito antes de `responses(x)` ser calculado para o mesmo x.
- `penalize` nunca produz valor negativo; `del` só ocorre em zero exato.
- A validação só é lida por `evaluate`; nunca passa por `train_one` ou `penalize`.

## 6. Hiperparâmetros e configuração

Todos escolhidos na validação; o teste não entra em nenhuma decisão.

| Parâmetro | Tipo | Padrão | Faixa sugerida | Observação |
| --- | --- | --- | --- | --- |
| rho0 | float | 0,25 | {0,1; 0,25; 0,5} | dureza inicial; o mais sensível junto com tau |
| tau | float | 0,1 | {0; 0,05; 0,1; 0,2} | margem mínima; tau = 0 desliga a tratativa de margem baixa |
| b | int | escolhido na validação | {1, 2, 3, 5, 10} | limiar fixo do treino; a inferência pode usar bleaching iterativo |
| decay | enum | inverse | constant / inverse / geometric | ρ\_t = ρ\_0, ρ\_0/(1+t) ou ρ\_0·γ^t |
| gamma | float | 0,8 | \[0,5; 0,95\] | só para decay = geometric |
| modulate | bool | false | — | ρ\_ef ∝ (R\_w − R\_y)/R\_w no erro |
| select\_error | enum | all | all / lowest\_count | endereços penalizados no erro |
| select\_margin | enum | closest\_to\_b | all / lowest\_count / closest\_to\_b | endereços penalizados na margem baixa |
| kappa | int | ⌈n/4⌉ | \[1, n\] | quantos endereços quando a seleção não é all |
| counter\_mode | enum | float | float / int\_stochastic | float no treino, arredondado para inteiro ao final; int\_stochastic é opcional |
| batch\_size | int | 64 | \[16, 512\] | afeta só a ordem das atualizações |
| E | int | 20 | \[5, 50\] | máximo de épocas |
| patience | int | 3 | \[2, 5\] | épocas sem melhora para early stopping |
| seed | int | 0 | — | única fonte de aleatoriedade |
|  |  |  |  |  |

Caso degenerado de verificação: `E = 1, rho0 = 0, tau = 0` deve reproduzir exatamente o treino one-shot.

## 7. Métricas e monitoramento por época

Todas saem da passada de classificação já feita no treino e na validação; nenhuma exige estrutura extra.

| Métrica | O que detecta | Como calcular | Sinal de alerta |
| --- | --- | --- | --- |
| acc\_val, C̄\_val | qualidade geral | acurácia e confiança média na validação em b fixo | acc cai 2 épocas seguidas → early stopping |
| acc por batch antigo vs. novo | esquecimento | acurácia nos primeiros 20% dos batches da época vs. nos últimos 20% | antiga cai enquanto nova sobe |
| flips | oscilação | fração de exemplos de treino cuja predição mudou em relação à época anterior | não decresce com t |
| n\_pen\_erro, n\_pen\_margem | intensidade da correção | contagem de penalizações de cada tratativa | n\_pen\_margem ≫ n\_pen\_erro sugere tau alto |
| massa Σm por discriminador | ρ agressivo | soma dos contadores de cada d | queda > 30% em uma época |
| endereços ativos por d | esparsidade | número de chaves com m > 0 | queda brusca em uma classe só |
| acc\_test | resultado final | uma vez, no melhor modelo | — |

Registro: uma linha por época com todas as colunas acima, mais ρ\_t e os hiperparâmetros fixos no cabeçalho do arquivo.

## 8. Protocolo experimental e critérios de aceite

Hipótese: o ganho sobre one-shot + bleaching aparece em datasets com classes sobrepostas, onde endereços impuros dominam a votação.

**Baseline, proposta e ablações**

- **Baseline**: WiSARD one-shot com bleaching iterativo (configuração atual). É a referência que a proposta precisa superar.
- **Proposta**: WiSARD por épocas completa (tratativa de erro + margem baixa).
- **Ablações** (medem a contribuição de cada parte da proposta):
  - só tratativa de erro (tau = 0);
  - só margem baixa (penalidade no erro desligada);
  - rho constante vs. com decaimento.
- Referência inferior opcional: WiSARD one-shot sem bleaching, para medir quanto o bleaching sozinho contribui.

**Procedimento**

1. Divisão estratificada 70 / 15 / 15 (treino / validação / teste), a mesma para baseline, proposta e ablações.
2. Para cada dataset, escolher b e o tamanho de endereço na validação do baseline; fixar para a proposta e as ablações.
3. Grid de rho0 e tau da seção 6 na validação; demais parâmetros no padrão.
4. Cinco sementes por configuração; reportar média e desvio.
5. Avaliar o teste uma vez por configuração, no melhor modelo da validação.

**Datasets sugeridos**

- MNIST e Fashion-MNIST binarizados (referência da literatura WiSARD; sobreposição moderada).
- Pelo menos um dataset tabular com classes desbalanceadas, para exercitar a normalização por N\_d.
- Um dataset sintético com sobreposição controlada entre classes, para confirmar a hipótese.

**Critérios de aceite**

- **CA-01** Acurácia de teste da proposta ≥ baseline em todos os datasets; melhora estatisticamente significativa (teste pareado sobre as 5 sementes) em pelo menos um com sobreposição.
- **CA-02** Nenhum caso de esquecimento: acurácia nos batches antigos não cai mais de 1 ponto percentual entre épocas consecutivas na configuração escolhida.
- **CA-03** flips decresce monotonicamente a partir da terceira época na configuração escolhida.
- **CA-04** O caso degenerado (E = 1, rho0 = 0, tau = 0) reproduz bit a bit a WiSARD one-shot.
- **CA-05** Tempo de treino da proposta ≤ E × tempo do baseline, com E o número de épocas efetivamente executadas.

## 9. API proposta sobre o wisardpkg

Extensão da classe Wisard do wisardpkg: uma função nova de treino e uma de leitura de respostas, implementadas no núcleo C++ e expostas nos bindings Python; a classe `Wisard` e `classify` não mudam. Nomes seguem o estilo do pacote.

```python
from dataclasses import dataclass

@dataclass
class EpochConfig:
    rho0: float = 0.25
    tau: float = 0.10
    b: int = 1
    decay: str = "inverse"          # constant | inverse | geometric
    gamma: float = 0.8
    modulate: bool = False
    select_error: str = "all"       # all | lowest_count
    select_margin: str = "closest_to_b"
    kappa: int | None = None        # None -> ceil(n / 4)
    counter_mode: str = "float"          # arredonda para int ao final do treino
    batch_size: int = 64
    E: int = 20
    patience: int = 3
    seed: int = 0


class Wisard:
    def responses(self, x, b: int) -> dict[str, int]:
        """R_d(x) para todo d em limiar b fixo. Não altera estado."""

    def penalize(self, d: str, addrs: list[tuple[int, int]], factor: float, rng) -> None:
        """m[d][r][a] *= factor para cada (r, a); remove chaves que zeram."""

    def train_epochs(self, X_train, y_train, X_val, y_val,
                     cfg: EpochConfig, log_path: str | None = None) -> "EpochReport":
        """Seção 5. Deixa self no melhor estado da validação. Retorna métricas por época."""


@dataclass
class EpochReport:
    best_epoch: int
    per_epoch: list[dict]           # uma linha por época, colunas da seção 7
    config: EpochConfig
```

Contrato de `train_epochs`:

- Pré-condição: `self` pode estar vazio ou já treinado one-shot; nos dois casos o laço começa da época 0.
- Pós-condição: `self` é um `Wisard` válido para `classify` e para o serializador existente.
- `responses` e `penalize` são públicas para permitir estratégias de seleção externas sem alterar o núcleo.

## 10. Casos de teste unitários

| ID | Cenário | Entrada | Resultado esperado |
| --- | --- | --- | --- |
| T-01 | Degenerado | E = 1, rho0 = 0, tau = 0, dataset qualquer | contadores idênticos ao treino one-shot, bit a bit |
| T-02 | Penalidade multiplicativa | m = 20, factor = 0,7, modo float | m = 14,0 |
| T-03 | Arredondamento estocástico | m = 1, factor = 0,7, 10 000 repetições | média em \[0,65; 0,75\]; valores só 0 ou 1 |
| T-04 | Nunca negativo | m = 1, factor = 0,01, 1 000 repetições | todo resultado ≥ 0 |
| T-05 | Remoção de chave | m = 1, factor = 0,3, rng força piso | chave removida da RAM |
| T-06 | Erro penaliza w | 2 classes, x classificado como w ≠ y | só endereços ativos de w diminuem; os de y aumentam em 1; demais inalterados |
| T-07 | Margem baixa penaliza l | w = y, C < tau | só endereços ativos de l diminuem; fator = 1 − rho\_ef·(tau − C)/tau |
| T-08 | Margem alta não penaliza | w = y, C ≥ tau | nenhum contador diminui |
| T-09 | Ordem classificar-antes-de-gravar | exemplo inédito, discriminador vazio | R\_y(x) = 0 na decisão; reforço só depois |
| T-10 | Seleção closest\_to\_b | ativos com m = {2, 50, 11, 300}, b = 10, kappa = 2 | penalizados os endereços com m = 11 e m = 2 |
| T-11 | Decaimento inverse | rho0 = 0,5, t = 0..3 | ρ\_t = 0,5; 0,25; 0,167; 0,125 |
| T-12 | Validação intocada | hash das RAMs antes e depois de evaluate(val) | idêntico |
| T-13 | Early stopping | acc\_val = \[0,8; 0,9; 0,85; 0,85; 0,85\], patience = 3 | para na época 4; modelo devolvido é o da época 1 |
| T-14 | Reprodutibilidade | duas execuções, mesma seed | contadores e log idênticos |
| T-15 | Fórmula k | m = 1000, b = 10, rho = 0,3 | 13 penalizações consecutivas sem reforço levam m abaixo de 10; 12 não |
| T-16 | Serialização | treino por épocas → salvar → carregar → classify | predições idênticas antes e depois |

## 11. Erratas dos slides e decisões em aberto

**Errata** No slide 2 (Regra de atualização), a penalidade do caso de margem baixa aparece como fator ρ(τ−C)/ρ; o fator correto, consistente com o esquema e com o RF-07, é:

```latex
m_{l,r}[a_r(x)] \leftarrow \Big(1 - \rho_t \,\frac{\tau - C}{\tau}\Big)\, m_{l,r}[a_r(x)]
```

**Decisões em aberto**

- [ ] Normalizar contadores por N\_d (b e tau como frações) já na primeira versão, ou só se o dataset desbalanceado mostrar necessidade?
- [ ] Padrão de `select_error`: todos os ativos (mais simples) ou `lowest_count` (mais protetor)?
- [ ] Representação do contador — decidido: float durante o treino, arredondado para inteiro ao final; a inferência opera sobre inteiros.
- [ ] Como o critério de parada do bleaching iterativo se comporta na inferência quando a distribuição dos contadores foi reajustada por épocas — medir na proposta.
- [ ] Posicionamento frente à ULEEN (épocas com straight-through estimator): confirmar que nenhuma etapa aqui usa gradiente, para sustentar a contribuição.
