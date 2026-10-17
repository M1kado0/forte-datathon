# Questions for organizers (P4 asks now). Each has our default if there is no answer

| # | Question | Default if no answer | Answer | Asked by / when |
|---|---|---|---|---|
| 1 | Когда жюри прогоняет тестовые видео через API и сколько часов сервис должен быть онлайн после 12:00? | Online from 11:00 until results are announced; tunnel laptop plugged in | | |
| 2 | Можно ли добавить в JSON свои поля (например `rule`, `confidence`) или схема строго фиксирована? | Keep the exact schema; put the rule citation inside `description`; extra fields only in UI-only endpoints | Схема фиксирована. Default holds | организаторы, 17.10 |
| 3 | Как считается таймкод для критерия 2: допуск в секундах или попадание в интервал события? | Onset of the above-threshold run for states, peak for actions (`docs/acceptance.md`, criterion 2) | На усмотрение команды. Default holds | организаторы, 17.10 |
| 4 | Запросы жюри идут последовательно или параллельно? | Assume 5 parallel; each job must finish in ≤ 60 s | | |
| 5 | Тип нарушения сравнивается со списком организаторов или оценивается человеком? Как называть типы? | Short Russian names, one per rule (`docs/TYPES.md`) | Ответ в брифе: закрытого списка нет, категории не раскрываются (FAQ 8), тип команда определяет сама (FAQ 2), `type` = «Название типа нарушения» (раздел 10). Отдаём `name` из `rules.yaml` | организаторы, 17.10 |
| 6 | Где и в каком виде сдавать (форма, репо, README)? | Form with UI URL + API URL + key; public GitHub repo with README | | |
| 8 | Можно ли развернуть сервис на своей арендованной VM (модели наши, внутри сервиса), или только на ноутбуке команды? | Laptop + cloudflared tunnel | | |
| 9 | Можно ли обучать на арендованной GPU (vast.ai) с обучающим датасетом? Инференс полностью локальный | Train only on team laptops | | |
| 7 | Можно ли ссылаться на открытые регламенты других банков РК? (FAQ 31 говорит да, подтвердить для типов без пункта НБРК) | Yes, marked as secondary source | | |
