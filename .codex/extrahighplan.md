# Быстрое резюме того, что уже переделано

- Монолитная логика из `TO_by_GA_and_RL_refinement_v3b.ipynb` вынесена в модульное ядро `gigala/topology/topology_optimiz/gen_rl/`.
- Добавлены модули `fem.py`, `metrics.py`, `representation.py`, `coarse_search.py`, `refine_env.py`, `pipeline.py` и `__init__.py`.
- Реализован `ProblemConfig`, кэшируемый `Evaluator.evaluate(mask, fidelity)`, `run_multistage_search(config) -> StageArtifacts` и `make_refine_env(...)`.
- Вместо прямого поиска по всей бинарной маске используется многостадийный бинарный pipeline `16x16 -> 32x32 -> 64x64`.
- На стадии search добавлены структурные мутации, boundary-only refinement, smart skipping, pruning isolated cells, smoothness/island heuristics и кэш FE-оценок.
- Ноутбук `TO_by_GA_and_RL_refinement_v3b.ipynb` переписан в orchestrator: он теперь создаёт конфиг, запускает pipeline, печатает метрики и визуализирует стадии.
- Добавлены тесты `tests/test_gen_rl_pipeline.py` на upsampling, cache parity, proxy/full ranking, pipeline artifacts и action masks.
- RL-ветка через `MaskablePPO` и image-like observation реализована интерфейсно, но локально не прогонялась, потому что в среде не установлены `torch`, `gymnasium`, `stable-baselines3`, `sb3-contrib`.

# Исходный подробный план

## Summary
- Текущий ноутбук не должен искать маску `64x64` как один плоский битовый вектор из `4096` переменных. Для `64x64` он становится orchestrator’ом над многостадийным бинарным пайплайном `16x16 -> 32x32 -> 64x64`.
- Основная идея: сохранить бинарность на всех этапах, но резко сократить число дорогих `64x64` FEM-оценок за счёт сжатого представления, boundary-only refinement, smart skipping, proxy-оценок и кэша.
- В план закладываются идеи из статьи и соседних экспериментов в репозитории: sparse rewards, one-island, smoothness, smart post-processing, anchor-node/multi-scale actions. Pseudo-3D для этой задачи не делаем основным путём.

## Key Changes
- Вынести тяжёлую логику из `TO_by_GA_and_RL_refinement_v3b.ipynb` в модули `gen_rl/`: `fem.py`, `metrics.py`, `representation.py`, `coarse_search.py`, `refine_env.py`, `pipeline.py`. Ноутбук оставить как точку запуска, визуализацию и сравнение стадий.
- Собрать единый `Evaluator` с предвычислением mesh connectivity, DOF maps, sparse pattern и boundary conditions для `16`, `32`, `64`. Интерфейс: `evaluate(mask, fidelity)` с режимами `proxy16`, `proxy32`, `full64`.
- Добавить обязательный кэш `hash(mask) -> EvalResult` и smart skipping в три ступени: геометрические фильтры, затем proxy-FEA, затем full `64x64` FEA только для прошедших кандидатов.
- Заменить текущий GA по всем ячейкам на бинарный coarse-stage search на `16x16`: population `128`, generations `250`, структурные мутации вместо случайного bit-flip. Использовать patch flip, edge erosion/dilation, volume-preserving swap, symmetry-aware crossover.
- После coarse stage брать top `8` кандидатов, апсемплить `16 -> 32` простым `2x2` block replication и запускать boundary-local search только в frontier band ширины `2`. Действия здесь бинарные `2x2` patch add/remove.
- После `32x32` брать top `2`, апсемплить `32 -> 64` тем же block replication и запускать финальный refinement только по frontier-ячейкам, а не по всей сетке.
- RL оставить только как финальную стадию на `64x64`, но полностью поменять постановку: вместо `MlpPolicy` и статического списка действий использовать `MaskablePPO` с маленькой `CNN` policy и action masking.
- В `64x64` env наблюдение сделать image-like, а не flat vector: каналы `occupancy`, `support/load mask`, `frontier band`. Действия: только валидные frontier edits, с макро-действиями `4x4 remove`, `2x2 remove`, `1x1 add/remove reconnect`.
- Горизонт эпизода ограничить `256` действиями. Full `64x64` FEM считать только каждые `16` шагов и в терминале; промежуточную reward-form использовать из proxy-метрик.
- Сохранить эвристики из статьи как first-class signal: sparse terminal reward, one-island bonus, smoothness penalty, isolated-cell pruning. Их использовать и в GA screening, и в RL reward, чтобы не тратить full FEA на заведомо плохие маски.
- Исправить текущую архитектурную проблему ноутбука: `args`, индексацию DOF, sparse structure и metric helpers не пересобирать внутри каждой fitness evaluation. Это должно жить в конфиге задачи и evaluator state.

## Public Interfaces
- `ProblemConfig(resolution, volume_target=0.55, load_case="cantilever", solver_backend, runtime_budget_hours=3)`
- `evaluate(mask: np.ndarray, fidelity: Literal["proxy16","proxy32","full64"]) -> EvalResult`
- `run_multistage_search(config) -> StageArtifacts`
- `make_refine_env(seed_mask, config) -> gym.Env`
- `StageArtifacts` должен содержать `coarse16`, `refined32`, `refined64`, `metrics`, `fea_counts`, `runtime`, чтобы ноутбук мог сравнивать стадии без повторного расчёта.

## Test Plan
- Проверить на unit-level, что апсемплинг `16 -> 32 -> 64` всегда сохраняет бинарность и корректно переносит supports/load.
- Проверить parity evaluator: cached/uncached full FEA дают одинаковый compliance; proxy-ranking на случайном наборе кандидатов коррелирует с full FEA достаточно хорошо для prefilter.
- Прогнать regression на существующем сценарии `25x25`/`32x32`: новый pipeline должен давать не хуже volume-feasible решения при меньшем числе full FEA, чем текущий notebook.
- Прогнать performance test на `64x64`: цель — end-to-end runtime `<= 3h`, общий budget full `64x64` FEM `<= 20k`, из них внутри RL `<= 5k`.
- Проверить RL env: action mask не выдаёт невалидных действий, disconnected/isolated маски либо чинятся post-processing, либо отбрасываются до full FEA.

## Assumptions
- Бинарность сохраняется на всех этапах; continuous-density/SIMP front-end не используется.
- Дополнительные зависимости разрешены. Базовый стек: `numpy`, `scipy`, `numba`, `torch`, `gymnasium`, `stable-baselines3`, `sb3-contrib`; solver backend с fallback на SciPy, если `cholmod`/другой быстрый sparse solver недоступен.
- Ноутбук остаётся главным артефактом для экспериментов, но вычислительное ядро живёт в `.py` модулях.
- Основная цель — сделать `64x64` practically computable в `1-3` часа, а не сохранить текущую форму ноутбука любой ценой.
