"""Sentences that interpret the batch-size and GroupNorm(1) results.

Each %%KEY%% placeholder in make_report.py is replaced here; a block or list item whose
placeholder has no text (results still missing) is dropped.
"""


def texts(main, bs, ana, fa):
    t = {}
    if 'gn1' in main:
        g, l, b = (main[n]['final_test_acc'] for n in ['gn1', 'ln', 'bn'])
        t['GN1_ANALYSIS'] = (
            f'GroupNorm(1) (نرمال‌سازی روی کل C,H,W هر نمونه) به {fa(g)}٪ رسید. این نتیجه فقط {fa(g - l, 1)} واحد بهتر از LayerNorm2d '
            f'و هنوز {fa(b - g, 1)} واحد پایین‌تر از BN است. پس فاصله‌ی LN با BN به نرمال‌سازی جداگانه‌ی هر پیکسل مربوط نیست. '
            'هر دو روشِ مستقل از batch در این آموزش کوتاه از BN عقب‌ترند. (هر آزمایش فقط با یک seed اجرا شد، پس تفاوت‌های زیر ۱ واحد قطعی نیستند.)')
    if bs:
        a = lambda n, B: bs[(n, B)]['final_test_acc']
        tm = lambda B: bs[('bn', B)]['epochs'][-1]['test_acc_trainmode']
        tr = lambda n, B: bs[(n, B)]['epochs'][-1]['train_acc']
        t['BS_SUMMARY'] = (
            f'برتری BN به batch بزرگ وابسته است. روی زیرمجموعه‌ی ۱۰ هزار تصویری، با batch برابر ۱۲۸ دقت BN برابر {fa(a("bn", 128))}٪ و دقت LN برابر {fa(a("ln", 128))}٪ بود. '
            f'با batch برابر ۴ دقت BN به {fa(a("bn", 4))}٪ رسید و LN با {fa(a("ln", 4))}٪ کمی بهتر شد.')
        t['BS_ANALYSIS'] = (
            f'**BN با کوچک شدن batch ضعیف می‌شود.** دقت BN با batch برابر ۱۲۸ و ۱۶ تقریباً برابر است ({fa(a("bn", 128))}٪ و {fa(a("bn", 16))}٪)، '
            f'ولی با batch برابر ۴ به {fa(a("bn", 4))}٪ می‌رسد ({fa(a("bn", 128) - a("bn", 4), 1)} واحد افت). '
            f'**LN اصلاً افت نمی‌کند**: {fa(a("ln", 128))}٪، {fa(a("ln", 16))}٪ و {fa(a("ln", 4))}٪. '
            'بهبود جزئی LN با batch کوچک‌تر به این دلیل است که در تعداد epoch ثابت، batch کوچک‌تر گام‌های به‌روزرسانی بیشتری دارد '
            '(۲۵۰۰ گام در هر epoch برای batch برابر ۴ در برابر ۷۸ گام برای batch برابر ۱۲۸). BN با وجود همین گام‌های بیشتر افت کرده است. '
            f'در نتیجه برتری {fa(a("bn", 128) - a("ln", 128), 1)} واحدی BN در batch برابر ۱۲۸ در batch برابر ۴ از بین می‌رود.'
            '@@'
            f'دو نشانه‌ی دیگر: (۱) دقت آموزش BN در batch برابر ۴ ({fa(tr("bn", 4))}٪) خیلی کمتر از دقت آزمون آن است. '
            'دلیلش این است که در آموزش، نرمال‌سازی با آمار پرنویز ۴ تصویر انجام می‌شود، ولی در آزمون با آمار running که پایدار است. '
            f'دقت آموزش LN ({fa(tr("ln", 4))}٪) با دقت آزمون آن هم‌خوان است. '
            f'(۲) اگر مدل BN در آزمون هم با batch برابر ۴ و آمار batch اجرا شود، دقت آن {fa(tm(4))}٪ است، در حالی که با batch برابر ۱۲۸ این عدد {fa(tm(128))}٪ است.'
            '@@'
            f'مدل بدون نرمال‌سازی در همین شرایط فقط به {fa(bs[("none", 128)]["final_test_acc"])}٪ رسید. پس هر دو نوع نرمال‌سازی برای این معماری ضروری‌اند. '
            'هزینه‌ی batch کوچک روی CPU هم زیاد است: هر epoch با batch برابر ۴ حدود ۴ برابر batch برابر ۱۲۸ طول کشید.'
            if ('none', 128) in bs else '')
        t['BS_CONCLUSION'] = (
            f'برتری BN به batch بزرگ وابسته است. با batch برابر ۴ دقت BN {fa(a("bn", 128) - a("bn", 4), 1)} واحد افت کرد و به LN رسید، '
            'در حالی که LN به اندازه‌ی batch حساس نبود. بدون نرمال‌سازی دقت بسیار پایین‌تر بود.')
    if t.get('BS_ANALYSIS'):
        for i, part in enumerate(t.pop('BS_ANALYSIS').split('@@')):
            t[f'BS_A{i + 1}'] = part.strip()
    return {k: v for k, v in t.items() if v}


def fill(blocks, main, bs, ana, fa):
    t = texts(main, bs, ana, fa)

    def sub(s):
        for k, v in t.items():
            s = s.replace(f'%%{k}%%', v)
        return s

    out = []
    for b in blocks:
        b = dict(b)
        if 'text' in b:
            b['text'] = sub(b['text'])
            if '%%' in b['text']:
                continue
        if 'items' in b:
            b['items'] = [x for x in (sub(i) for i in b['items']) if '%%' not in x]
        out.append(b)
    return out
