# Промпт для AI-агента Canva: правка слайдов питча

Вставляем блок ниже в агент Canva на открытой презентации. Тексты слайдов совпадают с `docs/PITCH.md`, числа берём из таблицы «Числа в питче» там же.

Если агент не принимает длинный запрос, отправляем его по частям в таком порядке: «Правила», «Слайды 1 to 3», «Слайды 4 to 6», «Приложение».

Canva работает в облаке. В презентацию не вставляем ни кадры, ни фрагменты, ни клипы из датасета (AGENTS.md, FAQ 6). Скриншот вставляем только интерфейса без кадров-доказательств или оставляем пустое место под живое демо.

```
You are editing an existing 16:9 presentation in Canva. It is a 5-minute pitch in Russian for a datathon. Keep the 6 main slides and the 4 appendix slides in the order given. Do not add slides, photos, stickers or icons. Change only what this brief asks.

RULES FOR EVERY SLIDE
1. All slide text and speaker notes are in Russian, exactly as quoted below, without the quote marks.
2. Use only the numbers given below. Write decimals with a comma, for example 5,0 с. Do not round, convert or add numbers.
3. Do not insert any frame, crop, screenshot or clip from a video. Use only text, the charts described below and the official logo.
4. White background. Magenta #A31551 is the accent colour. Dark Magenta #4A0221 and Coral #FC9385 appear only in the hero gradient on slide 1, running from Dark Magenta to Coral. Sky Blue #61CFFF is optional and used sparingly in charts. No other gradients, no purple or blue gradients, no frosted or translucent panels, no cards inside cards, no grey text on coloured backgrounds.
5. Fonts: Open Sans everywhere. Titles Open Sans Bold 700. Labels Open Sans SemiBold 600. Body Open Sans Regular 400. Playfair Display Regular 400 only for the slide 1 title. Handjet Medium 500 only for team names.
6. Logo: insert the official Forte logo file the team uploads (forte-logo.svg or forte-logo.png). Do not redraw, recolour, stretch or add effects to it.
7. Charts: one grey bar for the comparison and one magenta bar for the highlighted value. No rainbow colours. Build charts as native Canva charts from the numbers below, not as images.
8. No emoji. No icons above headings. No lorem ipsum. No stock photos.
9. Speaker notes: on each slide, write the timing given for that slide.
10. Never write these words or claims anywhere: "нарушение найдено", "доказано", "точность 100%", "нарушитель", "соответствует закону".

SLIDE 1 (speaker notes: "0:00 до 0:20")
- Background: hero gradient from Dark Magenta #4A0221 to Coral #FC9385.
- Forte logo, top left.
- Title, Playfair Display 400, white: "Проверка кассовой дисциплины по видео"
- Subtitle, Open Sans 400, white: "Сервис показывает проверяющему, в какую секунду смотреть, какой кадр это показывает и какой пункт НБРК подходит."
- Small line, Open Sans 600, white: "Команда {{название команды}} · Datathon '26". Leave the placeholder {{название команды}} as it is. The team fills it in.

SLIDE 2 (speaker notes: "0:20 до 1:40. Демо идёт вживую с ноутбука, результат готов до питча.")
- Title, Open Sans Bold: "Загружаем ролик и получаем таймкод, кадр и пункт НБРК"
- A large empty 16:9 rectangle with a thin magenta outline, centred. Inside it, a small grey label: "Живое демо на публичном адресе". Insert no screenshot here.
- Three labels to the right of the rectangle, each on its own line, Open Sans SemiBold 600:
  "1 Загружаем .mov"
  "2 Видим таймкод, кадр и описание"
  "3 Смотрим пункт НБРК"

SLIDE 3 (speaker notes: "1:40 до 2:40")
- Title, Open Sans Bold: "Таймкод по пику окон ближе к метке, чем по началу серии"
- Left side, big number, Open Sans Bold, magenta: "21 из 21". Label under it, Open Sans 400: "видео с нарушением получили признаки возможного нарушения".
- Right side, native bar chart. Chart title, Open Sans SemiBold: "Медиана ошибки таймкода, секунды". Two bars: "По началу серии" = 9,5 (grey); "По пику окон" = 5,0 (magenta).
- Small print under the chart, Open Sans 400, on white: "Проверка по видео, leave-one-video-out, 40 роликов. Порог подобран на тех же роликах."

SLIDE 4 (speaker notes: "2:40 до 3:30. По одному пункту на выдох.")
- Title, Open Sans Bold: "Пункты ТЗ, которые можно проверить в сервисе"
- Table with three columns: "Требование", "Где проверить", "Готово". The "Готово" cell gets a plain magenta check mark shape only for rows marked done in docs/acceptance.md at 06:00. Leave it empty for any other row. Do not use an emoji.
- Rows, in this order:
  1. "Загрузка видео, статус обработки, результат в интерфейсе" | "Публичный адрес"
  2. "Вердикт, таймкод, описание, кадры-доказательства" | "Ответ API, поле violations"
  3. "Пункт НБРК и нормативный источник" | "docs/RULES.md"
  4. "API по ключу X-API-Key, два шага: analyze и jobs" | "/api/v1/analyze, /api/v1/jobs"
  5. "Модели работают локально, облачных вызовов нет" | "README, docs/MODELS.md"
  6. "Работа без интернета после загрузки весов" | "make weights и offline-тест"

SLIDE 5 (speaker notes: "3:30 до 4:20")
- Title, Open Sans Bold: "Это подсказка, не вывод. Мы выбрали полноту"
- Left side, big number, Open Sans Bold, magenta: "18 из 19". Label under it, Open Sans 400: "чистых видео тоже получили признаки возможного нарушения (ложные срабатывания)".
- Right side, four short lines, Open Sans 400, one per line, no bullet icons:
  "Пропуск нарушения для банка недопустим, поэтому порог выбран на полноте."
  "Сервис не делает вывод там, где нужны невидимые данные, например проводка или сверка."
  "Медиана ошибки таймкода: 5,0 с."
  "Новый ракурс: замер [статус до питча]."
- Bottom line, Open Sans SemiBold 600, magenta: "Проверяющий открывает таймкод, а не весь ролик."

SLIDE 6 (speaker notes: "4:20 до 4:40")
- Title, Open Sans Bold, large: "Мы выбрали полноту: пропуск нарушения для банка недопустим."
- QR code generated in Canva, pointing to {{публичный адрес}}. Leave the placeholder text under it.
- Line, Open Sans SemiBold 600: "Попробуйте сами: загрузите свой ролик"
- Team names, Handjet Medium 500: "{{имена команды}}"

APPENDIX (not presented, used for questions). Titles Open Sans Bold, body Open Sans 400, no timing in the notes.
A1 title: "Данные: 40 роликов, 21 с нарушением и 19 без"
   Body: "Одна камера сверху, одна касса. Две расстановки стола, сейфа и кассира. Звука нет."
A2 title: "Проверка по видео, без утечки между обучением и проверкой"
   Body, one line each: "Leave-one-video-out: каждое видео целиком в обучении или в проверке."
   "Два дубля роликов объединены."
   "Найдено 21 из 21 видео с нарушением, 18 из 19 чистых с признаками."
   "Медиана ошибки таймкода: 5,0 с по пику, 9,5 с по началу серии."
A3 title: "Модели работают локально и не используют облако"
   Body, one line each: "DINOv2-small (Apache-2.0): признаки кадров, логистическая регрессия."
   "D-FINE-S (Apache-2.0): люди и объекты."
   "Описания берутся из шаблонов правил, модель текст не пишет."
A4 title: "Ограничения и известные ошибки"
   Body, one line each: "Ложные срабатывания: 18 из 19 чистых роликов."
   "Таймкод: медиана ошибки 5,0 с."
   "Новый ракурс: замер [статус до питча]."
   "Три типа нарушений без размеченных примеров."
   "Лица не распознаются, звук не используется."

When you finish, list every slide you changed, every placeholder in curly brackets or square brackets that you left in place, and any text you could not place. Do not invent text, numbers or images to fill a placeholder.
```
