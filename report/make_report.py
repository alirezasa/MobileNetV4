"""Build the report from results/*.json.

Writes
  report/report_blocks.json  -> rendered to Word by report/make_docx.js
  README.md                  -> the same report as Markdown (RTL)

The text lives here once; both outputs are generated from the same list of blocks.
Run from the repository root:  python report/make_report.py && node report/make_docx.js
"""
import json
import os

R = 'results'
FA_DIGITS = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')


def load(p):
    with open(p) as f:
        return json.load(f)


def opt(p):
    return load(p) if os.path.exists(p) else None


def fa(x, nd=2):
    """Number -> Persian digits (used in running text; tables keep Latin digits for alignment)."""
    s = f'{x:.{nd}f}' if isinstance(x, float) else str(x)
    return s.translate(FA_DIGITS).replace('.', '٫')


main = {n: opt(f'{R}/main_{n}.json') for n in ['bn', 'ln', 'gn1']}
main = {k: v for k, v in main.items() if v and 'final_test_acc' in v}  # finished runs only
ana = load(f'{R}/bn_vs_ln_analysis.json')
bs = {}
for n in ['bn', 'ln', 'none']:
    for b in [4, 16, 128]:
        h = opt(f'{R}/bs_{n}_{b}.json')
        if h and 'final_test_acc' in h:
            bs[(n, b)] = h
if not all((n, b) in bs for n in ['bn', 'ln'] for b in [4, 16, 128]):
    bs = {}  # the batch-size section is only written once every BN / LN run has finished

NAMES = {'bn': 'BatchNorm', 'ln': 'LayerNorm2d', 'gn1': 'GroupNorm(1)', 'none': 'بدون نرمال‌سازی'}
last = lambda h: h['epochs'][-1]
acc = {n: last(h)['test_acc'] for n, h in main.items()}
tracc = {n: last(h)['train_acc'] for n, h in main.items()}
sec = {n: sum(e['epoch_time'] for e in h['epochs']) / len(h['epochs']) for n, h in main.items()}

B = []  # blocks
add = B.append
H1 = lambda t: add({'t': 'h1', 'text': t})
H2 = lambda t: add({'t': 'h2', 'text': t})
P = lambda t: add({'t': 'p', 'text': t})
UL = lambda *items: add({'t': 'bullets', 'items': list(items)})
TABLE = lambda header, rows, widths=None, ltr=False: add({'t': 'table', 'header': header, 'rows': rows, 'widths': widths, 'ltr': ltr})
IMG = lambda path, caption, width=6.3: add({'t': 'img', 'path': path, 'caption': caption, 'width': width})
CODE = lambda *lines: add({'t': 'code', 'lines': list(lines)})
EQ = lambda t: add({'t': 'eq', 'text': t})

# ============================================================================ title
add({'t': 'title', 'text': 'پیاده‌سازی و آموزش شبکه‌ی ConvNeXt مبتنی بر بلوک‌های UIB معماری MobileNetV4 روی CIFAR-10'})
add({'t': 'subtitle', 'text': 'و بررسی تفاوت BatchNormalization و LayerNormalization'})
add({'t': 'meta', 'text': 'کد کامل، نتایج و نمودارها: github.com/alirezasa/MobileNetV4 (شاخه‌ی claude/nice-knuth-ot17ij)'})

# ============================================================================ 1. summary
H1('۱. خلاصه')
gap = acc['bn'] - acc['ln']
P(f'یک شبکه با ساختار کلی ConvNeXt ساخته شد که در آن هر بلوک، بلوک **Universal Inverted Bottleneck (UIB)** معماری MobileNetV4 است. '
  f'پیاده‌سازی UIB از کتابخانه‌ی **timm** گرفته شد. مدل ({fa(1.91)} میلیون پارامتر) با بهینه‌ساز **AdamW** به مدت ۳۰ epoch روی CIFAR-10 آموزش داده شد. '
  f'سپس همین شبکه یک بار با BatchNorm و یک بار با LayerNorm آموزش داده شد و چند آزمایش برای بررسی تفاوت این دو انجام شد.')
items = [
    f'دقت نهایی روی داده‌ی آزمون (۱۰٬۰۰۰ تصویر): **BatchNorm: {fa(acc["bn"])}٪** و **LayerNorm: {fa(acc["ln"])}٪**؛ '
    f'یعنی BN در این بودجه‌ی آموزشی کوتاه حدود {fa(gap, 1)} واحد درصد بهتر است.',
]
if 'gn1' in acc:
    items.append(f'GroupNorm با یک گروه (LN روی کل C,H,W هر نمونه): **{fa(acc["gn1"])}٪**.')
items += [
    'LayerNorm از ابتدا تا انتها کندتر همگرا شد و دقت آموزش آن نیز پایین‌تر ماند '
    f'({fa(tracc["ln"])}٪ در برابر {fa(tracc["bn"])}٪). پس فاصله از نوع underfitting است و نه overfitting.',
    'BN خروجی یک تصویر را به تصاویر دیگر batch وابسته می‌کند. اگر BN در زمان آزمون با آمار batch کار کند، '
    f'با batch آزمون ۲ دقت از {fa(acc["bn"])}٪ به {fa(ana["test_acc_running_vs_batch_stats"]["bn_batch_stats"]["2"])}٪ می‌رسد. LN در همه‌ی حالت‌ها دقیقاً یکسان است.',
    'BN در زمان inference به یک تبدیل خطی ثابت تبدیل می‌شود و در وزن کانولوشن قبلی ادغام می‌شود (۳۰ لایه از ۳۴ لایه‌ی BN حذف شد و دقت بدون تغییر ماند). LN این امکان را ندارد.',
]
if bs:
    items.append('%%BS_SUMMARY%%')
UL(*items)

# ============================================================================ 2. code
H1('۲. ساختار کد و نحوه‌ی اجرا')
TABLE(['فایل', 'توضیح'], [
    ['src/models.py', 'مدل ConvNeXt-UIB (بلوک UIB از timm) با لایه‌ی نرمال‌سازی قابل انتخاب'],
    ['src/data.py', 'بارگذاری CIFAR-10 در حافظه و augmentation برداری (برش تصادفی و قرینه‌ی افقی)'],
    ['train.py', 'حلقه‌ی آموزش: AdamW از timm، زمان‌بندی cosine با warmup، label smoothing، ثبت Loss و Accuracy در هر epoch'],
    ['run_experiments.sh', 'تمام اجراهای آموزشی این گزارش'],
    ['bn_vs_ln_analysis.py', 'آزمایش‌های BN و LN که به آموزش نیاز ندارند (روی مدل‌های آموزش‌دیده)'],
    ['plot_results.py', 'رسم نمودارها و جدول خلاصه'],
    ['MobileNetV4_UIB_CIFAR10.ipynb', 'نوت‌بوک قابل اجرا در Colab'],
    ['results/ , checkpoints/', 'تاریخچه‌ی هر epoch به صورت JSON، لاگ‌ها، نمودارها و وزن‌های آموزش‌دیده'],
], widths=[2.2, 4.3])
CODE('pip install -r requirements.txt',
     'python train.py --data ./data --norm bn --epochs 30 --out results/main_bn.json',
     'CIFAR_ROOT=./data ./run_experiments.sh',
     'python bn_vs_ln_analysis.py --data ./data && python plot_results.py')
P('همه‌ی اجراها روی یک CPU چهار هسته‌ای و بدون GPU انجام شد. به همین دلیل یک مدل کوچک و ۳۰ epoch انتخاب شد. '
  'روی GPU (مثلاً Colab) همین دستورات بدون تغییر اجرا می‌شوند.')

# ============================================================================ 3. UIB
H1('۳. بلوک UIB در MobileNetV4')
P('بلوک UIB در مقاله‌ی MobileNetV4 معرفی شد و تعمیم بلوک Inverted Bottleneck در MobileNetV2 است. '
  'هسته‌ی آن یک کانولوشن ۱×۱ برای افزایش کانال‌ها (expansion) و یک کانولوشن ۱×۱ برای کاهش آن‌ها (projection) است. '
  'دو کانولوشن depth-wise اختیاری هم دارد: یکی **قبل از** expansion و دیگری **بین** expansion و projection:')
CODE('x -> [DW kxk (start, optional)] -> PW expand (xE) + act -> [DW kxk (mid, optional)] + act -> PW project -> LayerScale -> DropPath -> (+x) -> y')
P('با روشن یا خاموش کردن این دو کانولوشن، چهار بلوک شناخته‌شده از یک کد واحد به دست می‌آیند:')
TABLE(['نوع', 'DW شروع', 'DW میانی', 'معادل'], [
    ['ConvNeXt-like', '✓', '✗', 'بلوک ConvNeXt: DW بزرگ روی کانال‌های کم، سپس MLP'],
    ['ExtraDW', '✓', '✓', 'نوآوری MobileNetV4: میدان دید بزرگ‌تر با هزینه‌ی کم'],
    ['IB', '✗', '✓', 'Inverted Bottleneck در MobileNetV2'],
    ['FFN', '✗', '✗', 'FFN در ترنسفورمرها (دو کانولوشن ۱×۱)'],
], widths=[1.4, 0.9, 0.9, 3.3])
P('در این تمرین از پیاده‌سازی timm استفاده شد: `timm.models._efficientnet_blocks.UniversalInvertedResidual`، یعنی همان کلاسی که مدل‌های '
  '`mobilenetv4_*` در timm با آن ساخته شده‌اند. در timm هر کانولوشن داخل UIB یک `ConvNormAct` است. '
  'پس برخلاف ConvNeXt اصلی که فقط یک LN بعد از DW دارد، **بعد از هر کانولوشن یک لایه‌ی نرمال‌سازی** هست. '
  'در نتیجه نوع نرمال‌سازی روی کل شبکه اثر زیادی دارد و این ساختار برای مقایسه‌ی BN و LN مناسب است.')

# ============================================================================ 4. architecture
H1('۴. معماری پیاده‌سازی‌شده: ConvNeXt-UIB')
P('ساختار کلی شبکه همان ConvNeXt است و هر بلوک ConvNeXt با یک UIB در حالت ConvNeXt-like '
  '(`dw_kernel_size_start=7`، `dw_kernel_size_mid=0`، `exp_ratio=4`) جایگزین شده است:')
TABLE(['بخش', 'جزئیات', 'خروجی'], [
    ['Stem', 'patchify: Conv 2×2 با stride ۲، سپس Norm (در ConvNeXt برای ImageNet این لایه 4×4/4 است)', '32×16×16'],
    ['Stage 1', '۲ بلوک UIB (DW 7×7، expansion ×4، GELU)', '32×16×16'],
    ['Stage 2', 'Downsample (Norm و Conv 2×2/2) و ۲ بلوک UIB', '64×8×8'],
    ['Stage 3', 'Downsample و ۴ بلوک UIB', '128×4×4'],
    ['Stage 4', 'Downsample و ۲ بلوک UIB', '256×2×2'],
    ['Head', 'Global Average Pooling، Norm و Linear(10)', '10'],
], widths=[1.0, 4.3, 1.2])
UL('تابع فعال‌سازی GELU (مانند ConvNeXt)، LayerScale با مقدار اولیه‌ی 1e-6، و Stochastic Depth (DropPath) با نرخ خطی تا ۰٫۱',
   f'تعداد پارامترها: {fa(1.907, 3)} میلیون (برای BN و LN یکسان)',
   'نوع نرمال‌سازی با آرگومان `norm` انتخاب می‌شود و در stem، همه‌ی UIBها، لایه‌های downsample و head یکسان است:',
   '`bn`: BatchNorm2d (و BatchNorm1d در head)؛ `ln`: LayerNorm2d از timm، یعنی LN روی بعد کانال در هر پیکسل (همان LN در ConvNeXt)؛ '
   '`gn1`: GroupNorm با یک گروه؛ `none`: بدون نرمال‌سازی')

# ============================================================================ 5. data + training
H1('۵. داده و حلقه‌ی آموزش')
H2('۵-۱. مجموعه‌داده‌ی CIFAR-10')
UL('۶۰٬۰۰۰ تصویر رنگی ۳۲×۳۲ در ۱۰ کلاس (هواپیما، خودرو، پرنده، گربه، گوزن، سگ، قورباغه، اسب، کشتی، کامیون)',
   '۵۰٬۰۰۰ تصویر برای آموزش و ۱۰٬۰۰۰ تصویر برای آزمون. از فایل اصلی cifar-10-python.tar.gz استفاده شد (md5 برابر با نسخه‌ی رسمی).',
   'augmentation: برش تصادفی ۳۲×۳۲ پس از padding چهار پیکسلی (reflect) و قرینه‌ی افقی با احتمال ۰٫۵. ورودی با میانگین و انحراف معیار کانال‌های CIFAR-10 نرمال شد.',
   'کل داده در حافظه نگه داشته می‌شود و augmentation به صورت برداری روی کل batch انجام می‌شود. با این کار به worker برای DataLoader نیازی نیست و همه‌ی هسته‌های CPU صرف مدل می‌شوند.')
H2('۵-۲. تنظیمات آموزش')
TABLE(['تنظیم', 'مقدار'], [
    ['بهینه‌ساز', 'AdamW با timm.optim.create_optimizer_v2 (بدون weight decay برای bias و پارامترهای نرمال‌سازی)'],
    ['نرخ یادگیری', '2e-3؛ warmup خطی ۲ epoch، سپس cosine تا 1e-6 (timm CosineLRScheduler، به‌روزرسانی در هر step)'],
    ['weight decay', '0.05'],
    ['batch size و تعداد epoch', '128 و 30 (۳۹۰ step در هر epoch)'],
    ['تابع هزینه', 'LabelSmoothingCrossEntropy(0.1) از timm در آموزش و cross-entropy ساده در ارزیابی'],
    ['معیارها', 'Loss و Accuracy آموزش و آزمون در هر epoch (timm.utils.accuracy و AverageMeter)'],
    ['قالب حافظه', 'channels_last: در CPU سرعت BN را حدود ۲ برابر کرد (510 به 251 میلی‌ثانیه در هر step)'],
], widths=[1.8, 4.7])
IMG('results/figures/lr_schedule.png', 'شکل ۱: نرخ یادگیری در طول آموزش (warmup و سپس cosine)', 3.6)

# ============================================================================ 6. experiment 1
H1('۶. آزمایش ۱: آموزش اصلی و روند Loss و Accuracy')
rows = []
for n, h in main.items():
    e = last(h)
    rows.append([NAMES[n], f'{e["test_acc"]:.2f}', f'{max(x["test_acc"] for x in h["epochs"]):.2f}', f'{e["train_acc"]:.2f}',
                 f'{e["test_loss"]:.3f}', f'{e["train_loss"]:.3f}', f'{sec[n]:.0f}'])
TABLE(['نرمال‌سازی', 'دقت آزمون نهایی', 'بهترین دقت آزمون', 'دقت آموزش', 'Loss آزمون', 'Loss آموزش', 'ثانیه در هر epoch'], rows,
      widths=[1.4, 0.9, 0.9, 0.85, 0.85, 0.85, 0.85])
IMG('results/figures/main_curves.png', 'شکل ۲: Loss و Accuracy آموزش (خط‌چین) و آزمون (خط پر) در هر epoch')
P('روند تغییرات در چند epoch منتخب:')
sel = [1, 2, 5, 10, 15, 20, 25, 30]
rows = []
for ep in sel:
    r = [str(ep)]
    for n in ['bn', 'ln']:
        e = main[n]['epochs'][ep - 1]
        r += [f'{e["train_loss"]:.3f}', f'{e["train_acc"]:.1f}', f'{e["test_loss"]:.3f}', f'{e["test_acc"]:.1f}']
    rows.append(r)
TABLE(['epoch', 'BN train loss', 'BN train acc', 'BN test loss', 'BN test acc', 'LN train loss', 'LN train acc', 'LN test loss', 'LN test acc'],
      rows, widths=[0.6] + [0.74] * 8, ltr=True)
H2('تحلیل')
an = [
    f'هر دو مدل بدون مشکل آموزش دیدند و Loss به صورت یکنواخت کم شد. افزایش دقت در epochهای پایانی با کاهش نرخ یادگیری در زمان‌بندی cosine هم‌زمان است.',
    f'BN از همان epoch اول جلوتر است ({fa(main["bn"]["epochs"][0]["test_acc"])}٪ در برابر {fa(main["ln"]["epochs"][0]["test_acc"])}٪) '
    f'و بیشترین فاصله حدود ۹ واحد بین epoch پنجم و دهم است. در پایان فاصله {fa(gap, 1)} واحد است.',
    f'دقت آموزش LN ({fa(tracc["ln"])}٪) هم کمتر از BN ({fa(tracc["bn"])}٪) است. پس LN در این بودجه‌ی ۳۰ epoch **کم‌برازش (underfit)** است و مشکل آن تعمیم‌پذیری نیست. '
    'این با تجربه‌ی ConvNeXt سازگار است: آن‌جا LN فقط با آموزش طولانی (۳۰۰ epoch) و augmentation قوی به دقت BN رسید یا کمی از آن بهتر شد.',
    'Loss آموزش از Loss آزمون بیشتر است، چون Loss آموزش شامل label smoothing (حداقل آن حدود ۰٫۵ است) و DropPath و augmentation است، ولی Loss آزمون cross-entropy ساده روی تصاویر بدون تغییر است.',
    'فاصله‌ی دقت آموزش و آزمون در BN حدود ۶ واحد و در LN حدود ۵ واحد است، یعنی overfitting شدیدی در هیچ‌کدام دیده نمی‌شود.',
]
if 'gn1' in acc:
    an.append('%%GN1_ANALYSIS%%')
an.append(f'زمان هر epoch روی CPU برای BN و LN تقریباً برابر است ({sec["bn"]:.0f} و {sec["ln"]:.0f} ثانیه). '
          'بدون channels_last آموزش با BN دو برابر کندتر بود، زیرا backward لایه‌ی BatchNorm در PyTorch روی CPU در قالب NCHW بهینه نیست. '
          'روی GPU معمولاً BN کم‌هزینه‌تر است.')
UL(*an)

# ============================================================================ 7. theory
H1('۷. تحقیق: تفاوت BatchNorm و LayerNorm')
H2('۷-۱. تعریف')
P('هر دو لایه یک کار را انجام می‌دهند: از ورودی میانگین کم می‌کنند، بر انحراف معیار تقسیم می‌کنند و سپس یک تبدیل خطی یادگرفتنی (γ و β) اعمال می‌کنند. '
  'تفاوت فقط در **مجموعه‌ای از مقادیر است که میانگین و واریانس روی آن حساب می‌شود** (S):')
EQ('x̂ = (x − μ_S) / √(σ²_S + ε)        y = γ · x̂ + β')
TABLE(['لایه', 'مجموعه‌ی آماری S برای tensor با ابعاد (N, C, H, W)', 'تعداد جفت (μ, σ²) برای N=8، C=16، H=W=6'], [
    ['BatchNorm2d', 'برای هر کانال، روی (N, H, W): همه‌ی نمونه‌های batch', '16'],
    ['LayerNorm2d (ConvNeXt)', 'برای هر نمونه و هر پیکسل، روی C', '288'],
    ['LayerNorm (ترنسفورمر)', 'برای هر نمونه یا توکن، روی بعد ویژگی‌ها', '—'],
    ['GroupNorm(1)', 'برای هر نمونه، روی (C, H, W)', '8'],
], widths=[1.6, 3.4, 1.5])
IMG('results/figures/norm_axes.png', 'شکل ۳: عناصری که میانگین و واریانس مشترک دارند (رنگی). فقط BN نمونه‌های مختلف batch را با هم ترکیب می‌کند.')
P('درستی این تعاریف در `bn_vs_ln_analysis.py` با مقایسه‌ی خروجی لایه‌های PyTorch و timm با فرمول دستی بررسی شد. '
  'بیشترین اختلاف حدود 5e-7 بود که همان خطای گرد کردن float32 است.')
H2('۷-۲. تفاوت‌های اصلی')
TABLE(['ویژگی', 'BatchNorm', 'LayerNorm'], [
    ['وابستگی به نمونه‌های دیگر batch', 'دارد: خروجی هر تصویر به بقیه‌ی تصاویر batch بستگی دارد', 'ندارد: هر نمونه مستقل نرمال می‌شود'],
    ['رفتار در آموزش و آزمون', 'متفاوت: آموزش با آمار batch، آزمون با میانگین متحرک (running mean/var)', 'یکسان'],
    ['batch کوچک', 'تخمین آماری پرنویز و افت دقت؛ batch با اندازه‌ی ۱ در head ممکن نیست', 'بدون مشکل'],
    ['آموزش توزیع‌شده', 'با batch کوچک روی هر GPU به SyncBatchNorm نیاز دارد', 'نیازی ندارد'],
    ['دنباله‌ها با طول متغیر (RNN، ترنسفورمر)', 'نامناسب', 'استاندارد'],
    ['هزینه‌ی inference', 'تقریباً صفر: در کانولوشن قبلی ادغام می‌شود', 'آمار به ورودی وابسته است و ادغام ممکن نیست'],
    ['اثر regularization', 'دارد (نویز آمار batch)', 'ندارد'],
    ['کاربرد رایج', 'CNNها: ResNet، EfficientNet، MobileNetV4', 'ترنسفورمرها، ViT، ConvNeXt'],
], widths=[1.8, 2.5, 2.2])
H2('۷-۳. چرا هر کدام کار می‌کنند؟')
UL('**BatchNorm** (Ioffe و Szegedy، ۲۰۱۵) با هدف کاهش «internal covariate shift» معرفی شد. پژوهش‌های بعدی (Santurkar و همکاران، ۲۰۱۸) نشان دادند '
   'که دلیل اصلی موفقیت آن بیشتر **هموارتر شدن سطح Loss** و پایداری گرادیان است. این ویژگی اجازه‌ی نرخ یادگیری بزرگ‌تر را می‌دهد. '
   'نویز آمار batch هم نوعی regularization است.',
   '**LayerNorm** (Ba و همکاران، ۲۰۱۶) برای RNNها معرفی شد، چون در آن‌ها آمار batch در طول زمان معنای ثابتی ندارد. بعد در ترنسفورمرها استاندارد شد. '
   'ConvNeXt (Liu و همکاران، ۲۰۲۲) نشان داد که در یک CNN مدرن، جایگزینی BN با LN مشکلی ایجاد نمی‌کند و دقت ImageNet را کمی بالا می‌برد (۸۱٫۴٪ به ۸۱٫۵٪).',
   '**MobileNetV4** با اینکه از ConvNeXt الهام گرفته، از BatchNorm استفاده می‌کند. هدف آن سرعت روی سخت‌افزار موبایل است و BN در زمان inference کاملاً در کانولوشن ادغام می‌شود.')

# ============================================================================ 8. experiment 2 (batch size)
H1('۸. آزمایش ۲: حساسیت به اندازه‌ی batch')
P('مهم‌ترین تفاوت عملی BN و LN وابستگی BN به اندازه‌ی batch است. برای بررسی آن، هر دو مدل با batch size برابر ۴، ۱۶ و ۱۲۸ آموزش داده شدند. '
  'به دلیل محدودیت CPU این آزمایش روی ۱۰٬۰۰۰ تصویر اول داده‌ی آموزش و برای ۱۰ epoch انجام شد. '
  'نرخ یادگیری با رابطه‌ی lr = 2e-3 · √(B/128) مقیاس شد و ارزیابی روی کل ۱۰٬۰۰۰ تصویر آزمون بود. '
  'برای BN دقت آزمون به دو روش گزارش شده است: با آمار running (حالت eval، روش معمول) و با آمار batch آزمون به اندازه‌ی batch آموزش.')
if bs:
    rows = []
    for b in [4, 16, 128]:
        for n in ['bn', 'ln', 'none']:
            if (n, b) in bs:
                h = bs[(n, b)]
                e = last(h)
                tm = e.get('test_acc_trainmode')
                rows.append([NAMES[n], str(b), f'{e["test_acc"]:.2f}', f'{tm:.2f}' if (tm is not None and n == 'bn') else '—',
                             f'{e["train_acc"]:.2f}', f'{e["test_loss"]:.3f}',
                             f'{sum(x["epoch_time"] for x in h["epochs"]) / len(h["epochs"]):.0f}'])
    TABLE(['نرمال‌سازی', 'batch', 'دقت آزمون (eval)', 'دقت آزمون با آمار batch', 'دقت آموزش', 'Loss آزمون', 'ثانیه در هر epoch'], rows,
          widths=[1.4, 0.6, 0.95, 1.1, 0.85, 0.8, 0.8])
    IMG('results/figures/batch_size.png', 'شکل ۴: چپ: دقت آزمون پس از ۱۰ epoch بر حسب batch size. راست: دقت در هر epoch برای کوچک‌ترین batch.')
    P('%%BS_A1%%')
    P('%%BS_A2%%')
    P('%%BS_A3%%')
else:
    P('(این آزمایش هنوز در حال اجراست.)')

# ============================================================================ 9. experiment 3 (analysis)
H1('۹. آزمایش ۳: تحلیل BN و LN روی مدل‌های آموزش‌دیده (بدون آموزش جدید)')
dep = ana['batch_dependence_rel_std']
H2('۹-۱. وابستگی خروجی یک تصویر به بقیه‌ی batch')
P('یک تصویر ثابت از داده‌ی آزمون، ۱۰ بار همراه با تصاویر تصادفی دیگر در یک batch قرار گرفت و مدل در حالت train (نرمال‌سازی با آمار batch) اجرا شد. '
  'نمودار، انحراف معیار نسبی logitهای همان یک تصویر در این ۱۰ اجرا را نشان می‌دهد.')
IMG('results/figures/batch_dependence.png', 'شکل ۵: تغییر خروجی یک تصویر ثابت با تغییر بقیه‌ی تصاویر batch', 4.6)
UL(f'در BN با batch برابر ۲، تغییرات logitها حدود {fa(dep["bn"]["2"] * 100, 0)}٪ اندازه‌ی خود آن‌هاست و کلاس پیش‌بینی‌شده در {fa(dep["bn_pred_changes"]["2"], 0)} اجرا از ۱۰ اجرا عوض شد. '
   f'با batch برابر ۲۵۶ این مقدار به {fa(dep["bn"]["256"] * 100, 0)}٪ کاهش می‌یابد ولی صفر نمی‌شود.',
   'در LN تغییرات دقیقاً صفر است: خروجی هر تصویر فقط به خود آن تصویر بستگی دارد.',
   'این همان نویزی است که در آموزش نقش regularization دارد. همچنین دلیل آن است که در زمان آزمون باید از آمار running استفاده کرد.')
H2('۹-۲. استفاده از آمار batch در زمان آزمون')
tb = ana['test_acc_running_vs_batch_stats']
IMG('results/figures/bn_test_batch_stats.png', 'شکل ۶: دقت آزمون مدل BN وقتی به جای آمار running از آمار batch آزمون استفاده شود', 4.6)
TABLE(['حالت', 'batch آزمون ۲', '۴', '۱۶', '۱۲۸', '۱۰۰۰'], [
    ['BN با آمار batch'] + [f'{tb["bn_batch_stats"][str(b)]:.2f}' for b in [2, 4, 16, 128, 1000]],
    ['BN با آمار running (eval)'] + [f'{tb["bn_running_stats"]:.2f}'] * 5,
    ['LN (حالت train یا eval)'] + [f'{tb["ln"]:.2f}'] * 5,
], widths=[2.0, 0.9, 0.9, 0.9, 0.9, 0.9])
P(f'اگر مدل BN در زمان آزمون با آمار batch اجرا شود، دقت با batch برابر ۲ از {fa(tb["bn_running_stats"])}٪ به {fa(tb["bn_batch_stats"]["2"])}٪ کاهش می‌یابد. '
  'پس قرار دادن مدل در حالت `model.eval()` برای BN ضروری است. این حالت در LN هیچ اثری ندارد. '
  f'همچنین BN در حالت train با یک تصویر (batch=1) اصلاً اجرا نمی‌شود: `{ana["train_mode_batch_size_1"]["bn"][:70]}...`')
H2('۹-۳. ادغام BN در کانولوشن (BN folding) و سرعت inference')
fo = ana['bn_folding']
lat = fo['latency_ms_batch32']
P('در حالت eval، BN یک تبدیل خطی ثابت برای هر کانال است: y = a·x + b. پس می‌توان آن را در وزن و bias کانولوشن قبلی ادغام کرد: '
  'W\' = W · γ/√(σ²+ε) و b\' = (b − μ) · γ/√(σ²+ε) + β. '
  f'با این کار {fo["bn2d_layers_before"] - fo["bn2d_layers_after"]} لایه از {fo["bn2d_layers_before"]} لایه‌ی BN مدل آموزش‌دیده حذف شد '
  f'(۴ لایه‌ی باقی‌مانده در stem و downsample هستند). بیشترین اختلاف logitها {fo["max_abs_diff_after_folding"]:.1e} بود و دقت آزمون دقیقاً {fa(fo["test_acc_folded"])}٪ باقی ماند. '
  'این ادغام برای LN ممکن نیست، چون μ و σ برای هر ورودی دوباره محاسبه می‌شوند.')
TABLE(['مدل', 'NCHW (ms)', 'channels_last (ms)'], [
    ['BN', f'{lat["bn_nchw"]:.1f}', f'{lat["bn_channels_last"]:.1f}'],
    ['BN پس از ادغام', f'{lat["bn_folded_nchw"]:.1f}', f'{lat["bn_folded_channels_last"]:.1f}'],
    ['LN', f'{lat["ln_nchw"]:.1f}', f'{lat["ln_channels_last"]:.1f}'],
], widths=[2.2, 2.1, 2.2])
P('زمان inference برای batch برابر ۳۲ روی CPU با ۲ thread اندازه‌گیری شد، هم‌زمان با اجرای آموزش دیگری روی هسته‌های دیگر. پس اعداد فقط برای مقایسه‌ی تقریبی هستند. '
  'با channels_last، مدل BN ادغام‌شده سریع‌ترین مدل است. روی سخت‌افزار موبایل و شتاب‌دهنده‌ها، که LN آن‌جا پشتیبانی بهینه ندارد، این تفاوت بیشتر است. '
  'به همین دلیل MobileNetV4 از BN استفاده می‌کند.')

# ============================================================================ 10. conclusion
H1('۱۰. جمع‌بندی')
UL('شبکه‌ای با ساختار ConvNeXt و بلوک‌های UIB معماری MobileNetV4 (پیاده‌سازی timm) ساخته شد و با AdamW و زمان‌بندی cosine روی CIFAR-10 آموزش داده شد. '
   f'با {fa(1.91)} میلیون پارامتر و ۳۰ epoch روی CPU به **{fa(acc["bn"])}٪** دقت آزمون رسید.',
   f'در این تنظیمات BatchNorm از LayerNorm بهتر بود ({fa(acc["bn"])}٪ در برابر {fa(acc["ln"])}٪). LN کندتر همگرا شد و در پایان آموزش هم کم‌برازش بود.',
   '%%BS_CONCLUSION%%',
   'LN مستقل از batch است و در آموزش و آزمون رفتار یکسان دارد. BN به batch وابسته است، دو رفتار متفاوت دارد و با batch کوچک ضعیف می‌شود، ولی هزینه‌ی inference آن تقریباً صفر است.',
   'پیشنهاد: برای CNNهای موبایل با batch معمولی BN انتخاب مناسب‌تری است (همان انتخاب MobileNetV4). وقتی batch کوچک است، داده‌ها دنباله‌ای‌اند یا رفتار یکسان در آموزش و آزمون مهم است، LN (یا GroupNorm) مناسب‌تر است.')

# ============================================================================ refs
H1('منابع')
refs = [
    'D. Qin et al., "MobileNetV4: Universal Models for the Mobile Ecosystem", ECCV 2024, arXiv:2404.10518',
    'Z. Liu et al., "A ConvNet for the 2020s" (ConvNeXt), CVPR 2022, arXiv:2201.03545',
    'S. Ioffe, C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", ICML 2015',
    'J. L. Ba, J. R. Kiros, G. E. Hinton, "Layer Normalization", arXiv:1607.06450, 2016',
    'Y. Wu, K. He, "Group Normalization", ECCV 2018',
    'S. Santurkar et al., "How Does Batch Normalization Help Optimization?", NeurIPS 2018',
    'I. Loshchilov, F. Hutter, "Decoupled Weight Decay Regularization" (AdamW), ICLR 2019',
    'R. Wightman, "PyTorch Image Models (timm)", github.com/huggingface/pytorch-image-models',
    'A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images" (CIFAR-10), 2009',
]
add({'t': 'refs', 'items': refs})

# ============================================================================ fill the result-dependent sentences
from fill_text import fill  # noqa: E402  (sentences that depend on the batch-size / GroupNorm results)

B = fill(B, main=main, bs=bs, ana=ana, fa=fa)

os.makedirs('report', exist_ok=True)
with open('report/report_blocks.json', 'w') as f:
    json.dump(B, f, ensure_ascii=False, indent=0)


# ============================================================================ markdown
def md_inline(t):
    return t


def to_md(blocks):
    out = ['<div dir="rtl">', '']
    fig_open = False
    for b in blocks:
        t = b['t']
        if t == 'title':
            out += [f'# {b["text"]}', '']
        elif t == 'subtitle':
            out += [f'### {b["text"]}', '']
        elif t == 'meta':
            out += [f'*{b["text"]}*', '']
        elif t == 'h1':
            out += [f'## {b["text"]}', '']
        elif t == 'h2':
            out += [f'### {b["text"]}', '']
        elif t == 'p':
            out += [b['text'], '']
        elif t == 'bullets':
            out += [f'- {i}' for i in b['items']] + ['']
        elif t == 'refs':
            out += [f'{k + 1}. {i}' for k, i in enumerate(b['items'])] + ['']
        elif t == 'table':
            esc = lambda c: str(c).replace('|', '\\|')
            out += ['| ' + ' | '.join(map(esc, b['header'])) + ' |', '|' + '---|' * len(b['header'])]
            out += ['| ' + ' | '.join(map(esc, r)) + ' |' for r in b['rows']] + ['']
        elif t == 'img':
            out += [f'![{b["caption"]}]({b["path"]})', '', f'*{b["caption"]}*', '']
        elif t == 'code':
            out += ['</div>', '', '```', *b['lines'], '```', '', '<div dir="rtl">', '']
        elif t == 'eq':
            out += ['</div>', '', '```', b['text'], '```', '', '<div dir="rtl">', '']
    out += ['</div>', '']
    return '\n'.join(out)


with open('README.md', 'w') as f:
    f.write(to_md(B))
print(f'{len(B)} blocks -> report/report_blocks.json, README.md')
