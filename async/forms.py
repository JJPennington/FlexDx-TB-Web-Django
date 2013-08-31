from django import forms

class ModelForm(forms.Form):
    type_select = forms.ChoiceField(choices = ( (0, 'Single Strategy [Click for List]'),
                                                (1, 'All Strategies') ), widget=forms.RadioSelect() )
    int_select = forms.ChoiceField(choices=( (0, '1. Smear'),
                                (1, '2. Culture for retreatment'),
                                (7, '3. GeneXpert for HIV positive only'),
                                (6, '4. GeneXpert for smear positive only'),
                                (2, '5. GeneXpert for all'),
                                (8, '6. GeneXpert for all, culture confirmed'),
                                (3, '7. MODS/TLA'),
                                (4, '8. Same-day smear microscopy'),
                                (5, '9. Same-day GeneXpert') ), widget=forms.RadioSelect() )
    t_inc = forms.FloatField(
            label='Target TB incidence, per 100,000',
            max_value=100000, min_value=0.0, initial=250, widget=forms.TextInput(attrs={'size': '2'}))
    t_mdr = forms.FloatField(
            label='Target MDR-TB prevalence among new cases, %',
            max_value=100.0, min_value=0.0, initial=3.7, widget=forms.TextInput(attrs={'size': '2'}))
    t_hiv = forms.FloatField(
            label='Target adult HIV prevalence, %',
            max_value=100.0, min_value=0.0, initial=0.83, widget=forms.TextInput(attrs={'size': '2'}))
    t_drug1_cost = forms.FloatField(
            label='Treatment of one patient with first-line drugs, $',
            min_value=0.0, initial=500, widget=forms.TextInput(attrs={'size': '2'}))
    #Added 1 August 2013
    t_sm_cost = forms.FloatField(
            label='Full sputum smear evaluation (e..g, collection & evaluation of 2 smears), $', min_value=0.0, initial=2,
            widget=forms.TextInput(attrs={'size':'2'}))
    t_gxp_cost = forms.FloatField(
            label='Single Xpert MTB/RIF test, $', min_value=0.0, initial=15,
            widget=forms.TextInput(attrs={'size':'2'}))
    t_cx_cost = forms.FloatField(
            label='Single automated liquid-media culture (MGIT) without DST, $', min_value=0.0, initial=20,
            widget=forms.TextInput(attrs={'size':'2'}))
    t_mods_cost = forms.FloatField(
            label='Single microcolony-based culture (MODS or thin-layer agar), $', min_value=0.0, initial=5,
            widget=forms.TextInput(attrs={'size':'2'}))
    t_dst_cost = forms.FloatField(
            label='Single automated liquid-media culture (MGIT) with DST, $', min_value=0.0,
            initial=40, widget=forms.TextInput(attrs={'size':'2'}))
    t_sd_cost = forms.FloatField(
            label='Full sputum smear, including extra costs to make results available same day, $', min_value=0.0,
            initial=10, widget=forms.TextInput(attrs={'size':'2'}))
    t_sdgxp_cost = forms.FloatField(
            label='Single Xpert, including extra costs to make results available same day, $', min_value=0.0,
            initial=30, widget=forms.TextInput(attrs={'size':'2'}))
    t_drug2_cost = forms.FloatField(
            label ='Treatment of one patient with retreatment ("category 2") regimen, $', min_value=0.0,
            initial=1000, widget=forms.TextInput(attrs={'size':'2'}))
    t_drug3_cost = forms.FloatField(
            label ='Treatment of one patient with second-line (MDR) drugs, $', min_value=0.0,
            initial=5000, widget=forms.TextInput(attrs={'size':'2'}))
    t_outpt_cost = forms.FloatField(
            label ='One outpatient visit (e.g., for TB diagnosis), $', min_value=0.0,
            initial=10, widget=forms.TextInput(attrs={'size':'2'}))


