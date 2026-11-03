from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,csv,textwrap,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib.ticker import MultipleLocator

import os
ROOT=Path(os.environ.get("METEO_WORK",Path(__file__).parent))
raw=json.loads((ROOT/'respuesta.json').read_text());query=json.loads((ROOT/'consulta.json').read_text())
h=raw['hourly'];times=[datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in h['time']]
consulted=datetime.fromisoformat(query['consulted_at_utc']);now=consulted.astimezone(timezone.utc)
keys=lambda prefix: sorted([k for k in h if k==prefix or k.startswith(prefix+'_member')])
tkeys=keys('temperature_2m');rkeys=keys('precipitation')
assert len(tkeys)==len(rkeys)==51
assert [k.removeprefix('temperature_2m') for k in tkeys]==[k.removeprefix('precipitation') for k in rkeys]
T=np.array([h[k] for k in tkeys],float);R=np.array([h[k] for k in rkeys],float)
assert T.shape==R.shape
wkeys=keys('wind_speed_10m');dkeys=keys('wind_direction_10m')
assert len(wkeys)==len(dkeys)==51
W=np.array([h[k] for k in wkeys],float);D=np.array([h[k] for k in dkeys],float)
assert h and raw['hourly_units'][wkeys[0]]=='km/h'
complete=np.isfinite(T).all(axis=0)&np.isfinite(R).all(axis=0)&np.isfinite(W).all(axis=0)&np.isfinite(D).all(axis=0)
neg=int((R<0).sum());minimum=float(np.nanmin(R));R=np.maximum(R,0)
# Inicio en la siguiente hora UTC múltiplo de tres: solo periodos futuros completos.
start=now.replace(hour=now.hour//3*3,minute=0,second=0,microsecond=0)
if start<now:start+=timedelta(hours=3)
inds=[i for i,t in enumerate(times) if t>=start and t.hour%3==0 and complete[i]]
assert len(inds)>=2, "Cobertura insuficiente"
xt=[times[i] for i in inds];temp=T[:,inds]
# La precipitación horaria corresponde a la hora anterior; cada bloque suma sus 3 horas.
ends=[i for i in inds[1:] if i>=2 and complete[i-2:i+1].all()]
assert all((times[b]-times[a]).total_seconds()==10800 for a,b in zip(inds,inds[1:]))
assert np.isfinite(T[:,inds]).all()
xr=[times[i] for i in ends];rain=np.array([R[:,i-2:i+1].sum(axis=1) for i in ends]).T
assert np.isfinite(rain).all() and np.min(rain)>=0
percentiles=[10,25,50,75,90]
qt=np.percentile(temp,percentiles,axis=0);qr=np.percentile(rain,percentiles,axis=0)
qw=np.percentile(W[:,inds],percentiles,axis=0)
pop=(rain>=.2).mean(axis=0)*100
width=qt[4]-qt[0]
first=np.mean(width[np.array([(t-start).total_seconds()<72*3600 for t in xt])]);last=np.mean(width[-16:])
peak=int(pop.argmax());date_peak=xr[peak]
early=np.array([(t-start).total_seconds()<72*3600 for t in xr]);late=~early
if last>first*1.25:
    temperature_text='Los escenarios muestran mayor acuerdo en las temperaturas durante los primeros tres días y más diferencias hacia el final del periodo.'
elif last<first*.8:
    temperature_text='Los escenarios muestran menos diferencias de temperatura al final que durante los primeros tres días; la dispersión no aumenta de forma continua con el plazo.'
else:
    temperature_text='La dispersión de temperaturas es parecida al principio y al final del periodo; las bandas muestran los momentos con mayor desacuerdo.'
early_peak=int(np.argmax(np.where(early,pop,-1)));late_peak=int(np.argmax(np.where(late,pop,-1))) if late.any() else None
rain_text=(f'La lluvia presenta su mayor coincidencia inicial en un bloque que termina el {xr[early_peak]:%d/%m} a las {xr[early_peak]:%H} UTC: '
           f'{int(round(pop[early_peak]))} % de los escenarios prevé al menos 0,2 mm en tres horas. ')
if late_peak is not None and pop[late_peak]>=50:
    rain_text+=f'También aparece una señal lluviosa hacia el {xr[late_peak]:%d/%m}, con diferencias en cantidad y horario. '
else:
    rain_text+='En el resto del plazo hay menos coincidencia en lluvia. '
wi=int(qw[2].argmax()); spread=qw[4,wi]-qw[0,wi]
wind_text=(f'El viento a 10 m alcanza su mayor mediana el {xt[wi]:%d/%m %H} UTC: {qw[2,wi]:.0f} km/h, con el 80 % central entre {qw[0,wi]:.0f} y {qw[4,wi]:.0f} km/h. ')
sectors=((D[:,inds]+22.5)%360//45).astype(int);calm=W[:,inds]<1
dominant_counts=np.bincount(sectors[~calm],minlength=8); k=int(dominant_counts.argmax()); frac=dominant_counts[k]/max(1,dominant_counts.sum())
direction_text=f'En el conjunto del periodo, el sector más frecuente es {['norte','noreste','este','sureste','sur','suroeste','oeste','noroeste'][k]} ({frac:.0%} de los casos sin calma); el panel muestra los cambios horarios. '
# Indicadores descriptivos del acuerdo, sin calibración de acierto.
dates=sorted(set(t.date() for t in xt))
labels=['Temperatura','Dirección del viento','Velocidad del viento','Cantidad de lluvia / 3 h','¿Lloverá en cada bloque?']
def level_width(value,high,medium):
 return 'Mayor' if value<=high else 'Intermedio' if value<=medium else 'Poco'
def level_fraction(value):
 return 'Mayor' if value>=80 else 'Intermedio' if value>=60 else 'Poco'
table=[[] for _ in labels];evidence=[]
for day in dates:
 it=np.array([t.date()==day for t in xt]);ir=np.array([t.date()==day for t in xr])
 direction_fractions=np.array([((sectors==k)&~calm).mean(axis=0)*100 for k in range(8)]+[calm.mean(axis=0)*100])
 tw=float(np.max(qt[4,it]-qt[0,it]));ww=float(np.max(qw[4,it]-qw[0,it]))
 dc=float(np.min(direction_fractions[:,it].max(axis=0)))
 rw=float(np.max(qr[4,ir]-qr[0,ir])) if ir.any() else None
 oc=float(np.min(np.maximum(pop[ir],100-pop[ir]))) if ir.any() else None
 values=[level_width(tw,2,4),level_fraction(dc),level_width(ww,5,10),level_width(rw,1,5) if rw is not None else 'Sin datos',level_fraction(oc) if oc is not None else 'Sin datos']
 for row,value in zip(table,values):row.append(value)
 evidence.append({'date_utc':str(day),'temperature_max_p10_p90_width_c':tw,'wind_max_p10_p90_width_kmh':ww,'direction_min_largest_sector_percent':dc,'rain_max_p10_p90_width_mm_3h':rw,'rain_occurrence_min_agreement_percent':oc,'agreement':dict(zip(labels,values))})
def strongest_doubt(row):
 ranks={'Mayor':0,'Intermedio':1,'Poco':2,'Sin datos':3};worst=max(ranks[x] for x in row)
 ds=', '.join(d.strftime('%d/%m') for d,x in zip(dates,row) if ranks[x]==worst)
 return row[next(i for i,x in enumerate(row) if ranks[x]==worst)],ds
sentences=[]
for label,row in zip(labels,table):
 lev,ds=strongest_doubt(row)
 if lev=='Mayor':txt='Los escenarios coinciden bastante en todas las horas disponibles.'
 else:txt=f"{'Hay más diferencias' if lev=='Poco' else 'Hay algunas diferencias'} el {ds}. Consulta la tabla para los demás días."
 sentences.append(label+': '+txt)
sentences[3]+=' Puede haber acuerdo en que lloverá y dudas sobre cuánto caerá.'
sentences[4]='¿Lloverá?: el porcentaje de las barras indica cuántos escenarios dan lluvia; las barras bajas indican que la mayoría no la da.'
summary='\n'.join(sentences)
(ROOT/'acuerdo_por_dia.json').write_text(json.dumps({'method':'Reglas descriptivas; peor bloque disponible de cada día; no calibrado','days':evidence},ensure_ascii=False,indent=2))
statistics={'consulted_at_utc':query['consulted_at_utc'],'model':'ECMWF IFS 9 km · O1280 Europe Ensemble','members':51,'model_id':query['model'],'available_to_utc':times[np.where(complete)[0][-1]].isoformat(),'requested_location':{'lat':36.17,'lon':-5.35},'returned_grid':{k:raw[k] for k in ['latitude','longitude','elevation']},'model_run':'not_provided_by_response','start_utc':start.isoformat(),'end_utc':xt[-1].isoformat(),'first_72h_mean_p10_p90_width_c':float(first),'last_48h_mean_p10_p90_width_c':float(last),'rain_threshold_mm_3h':.2,'maximum_member_fraction_percent':float(pop[peak]),'maximum_fraction_block_end_utc':date_peak.isoformat(),'negative_hourly_precipitation_values_clamped_to_zero':neg,'raw_minimum_precipitation_mm':minimum,'summary':summary,'response_sha256':hashlib.sha256((ROOT/'respuesta.json').read_bytes()).hexdigest()}
(ROOT/'estadisticas.json').write_text(json.dumps(statistics,ensure_ascii=False,indent=2))
with (ROOT/'percentiles_3h.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['valid_utc']+[f'temperature_p{p}_c' for p in percentiles]+[f'wind_speed_p{p}_kmh' for p in percentiles]+[f'rain_last3h_p{p}_mm' for p in percentiles]+['fraction_members_rain_ge_0.2mm_last3h_percent'])
 for j,i in enumerate(inds):
  ri=ends.index(i) if i in ends else None
  w.writerow([times[i].isoformat()]+qt[:,j].tolist()+qw[:,j].tolist()+(['']*6 if ri is None else qr[:,ri].tolist()+[float(pop[ri])]))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelcolor':'#473b2a','xtick.color':'#615740','ytick.color':'#615740'})
fig=plt.figure(figsize=(18,24),facecolor='#fff8df')
gs=fig.add_gridspec(5,1,height_ratios=[2,1.4,1.8,1.8,1.7],left=.065,right=.965,bottom=.115,top=.63,hspace=.32)
axes=[fig.add_subplot(gs[0])]
axes+=[fig.add_subplot(gs[i],sharex=axes[0]) for i in [1,2,3,4]]
for ax in axes:
 ax.set_facecolor('#fffdf6');ax.set_axisbelow(True);ax.grid(axis='y',color='#d8c9a2',alpha=.65,lw=.65)
 ax.xaxis.set_major_locator(md.DayLocator(tz=timezone.utc));ax.xaxis.set_minor_locator(md.HourLocator(byhour=[6,12,18],tz=timezone.utc))
 ax.grid(axis='x',which='major',color='#c2a667',alpha=.7,lw=.8);ax.grid(axis='x',which='minor',color='#ddcda9',alpha=.5,lw=.4)
 for spine in ax.spines.values():spine.set_color('#c7b98e')
 ax.set_xlim(start,xt[-1]);ax.tick_params(axis='x',which='both',labelbottom=False)
 # Fines de semana, como referencia de calendario, por debajo de las bandas.
 day=start.replace(hour=0)
 while day<xt[-1]:
  if day.weekday()>=5:ax.axvspan(day,day+timedelta(days=1),color='#d4dce0',alpha=.22,zorder=0)
  day+=timedelta(days=1)
ax=axes[0]
for row in temp:ax.plot(xt,row,color='#b18e71',alpha=.13,lw=.6,zorder=1)
ax.fill_between(xt,qt[0],qt[4],color='#f0bf83',alpha=.67,zorder=2,label='80 % central · P10–P90')
ax.fill_between(xt,qt[1],qt[3],color='#dc8b4e',alpha=.74,zorder=3,label='50 % central · P25–P75')
ax.plot(xt,qt[2],color='#96342d',lw=2,zorder=4,label='Mediana · P50')
ax.set_ylabel('Temperatura a 2 m (°C)');ax.yaxis.set_major_locator(MultipleLocator(2));ax.set_title('TEMPERATURA · CUANTO MÁS ANCHA LA BANDA, MAYOR DISPERSIÓN',loc='left',fontsize=12,weight='bold',color='#473b2a',pad=9)
ax.legend(loc='upper right',ncol=3,fontsize=9,frameon=True,facecolor='#fffdf6')
ax=axes[3]
# Los intervalos son acumulaciones completas de tres horas: representación escalonada.
xstep=[t-timedelta(hours=3) for t in xr]+[xr[-1]]
for k in [0,1,2,3,4]:assert len(qr[k])==len(xr)
ax.fill_between(xstep,np.r_[qr[0],qr[0,-1]],np.r_[qr[4],qr[4,-1]],step='post',color='#b6d9e7',alpha=.9,label='80 % central · P10–P90')
ax.fill_between(xstep,np.r_[qr[1],qr[1,-1]],np.r_[qr[3],qr[3,-1]],step='post',color='#70afc5',alpha=.9,label='50 % central · P25–P75')
ax.stairs(qr[2],xstep,color='#155b80',lw=1.8,label='Mediana · P50')
ax.set_ylim(0,max(1,float(qr[4].max())*1.25));ax.set_ylabel('Precipitación (mm / 3 h)');ax.set_title('LLUVIA · ACUMULACIÓN EN CADA BLOQUE DE TRES HORAS',loc='left',fontsize=12,weight='bold',color='#473b2a',pad=9)
ax.legend(loc='upper right',ncol=3,fontsize=9,frameon=True,facecolor='#fffdf6')
ax=axes[2]
ax.fill_between(xt,qw[0],qw[4],color='#c8c4e8',alpha=.85,label='80 % central · P10–P90')
ax.fill_between(xt,qw[1],qw[3],color='#938ac8',alpha=.8,label='50 % central · P25–P75')
ax.plot(xt,qw[2],color='#51438d',lw=2,label='Mediana · P50')
ax.set_ylim(0,max(10,float(qw[4].max())*1.2));ax.set_ylabel('Velocidad a 10 m (km/h)')
ax.set_title('VIENTO · VELOCIDAD A 10 m Y DISPERSIÓN ENTRE ESCENARIOS',loc='left',fontsize=12,weight='bold',color='#473b2a',pad=9)
ax.legend(loc='upper right',ncol=3,fontsize=9,frameon=True,facecolor='#fffdf6')
# Direction uses categorical fractions, avoiding arithmetic averaging of angles.
sector_names=['N','NE','E','SE','S','SO','O','NO']
direction=D[:,inds];speed=W[:,inds]
sectors=((direction+22.5)%360//45).astype(int)
calm=speed<1
fractions=np.array([((sectors==k)&~calm).mean(axis=0)*100 for k in range(8)]+[calm.mean(axis=0)*100])
assert np.allclose(fractions.sum(axis=0),100)
ax=axes[1]
xedges=np.array([md.date2num(t) for t in xt]);xedges=np.r_[xedges-1.5/24,xedges[-1]+1.5/24]
mesh=ax.pcolormesh(xedges,np.arange(10)-.5,fractions,cmap='YlGnBu',vmin=0,vmax=100,shading='flat')
ax.set_yticks(range(9),sector_names+['Calma']);ax.set_ylim(8.5,-.5)
ax.set_ylabel('Procedencia')
ax.set_title('DIRECCIÓN DEL VIENTO · PORCENTAJE DE ESCENARIOS EN CADA SECTOR',loc='left',fontsize=12,weight='bold',color='#473b2a',pad=9)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
cax=inset_axes(ax,width='17%',height='7%',loc='upper right',borderpad=.8)
cb=fig.colorbar(mesh,cax=cax,orientation='horizontal',ticks=[0,50,100]);cb.ax.tick_params(labelsize=7,pad=1);cb.ax.set_title('Escenarios (%)',fontsize=7,pad=3)
ax=axes[4]
ax.bar([t-timedelta(hours=1.5) for t in xr],pop,width=3/24*.85,color='#279ca0',edgecolor='none')
ax.set_ylim(0,100);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Escenarios (%)')
ax.set_title('ESCENARIOS CON LLUVIA ≥0,2 mm EN TRES HORAS',loc='left',fontsize=12,weight='bold',color='#473b2a',pad=9)
with (ROOT/'escenarios_direccion_lluvia.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['valid_utc']+['direction_'+v+'_percent' for v in sector_names+['calm']]+['fraction_members_rain_ge_0.2mm_last3h_percent'])
 for j,i in enumerate(inds):
  ri=ends.index(i) if i in ends else None
  w.writerow([times[i].isoformat()]+fractions[:,j].tolist()+([''] if ri is None else [float(pop[ri])]))
days_es=['lun','mar','mié','jue','vie','sáb','dom']
from matplotlib.ticker import FuncFormatter
ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:days_es[md.num2date(v,tz=timezone.utc).weekday()]+'\n'+md.num2date(v,tz=timezone.utc).strftime('%d/%m')))
ax.tick_params(axis='x',which='major',labelbottom=True,pad=7);ax.set_xlabel('Fechas y horas en UTC · marcas secundarias cada seis horas',labelpad=7)
fig.text(.065,.959,'¿QUÉ CONFIANZA TIENE LA PREVISIÓN?',fontsize=26,weight='bold',color='#473b2a')
fig.text(.065,.927,'La Línea de la Concepción · Meteograma de incertidumbre',fontsize=18,color='#66583d')
fig.text(.065,.901,f'ECMWF IFS ENS · 51 escenarios · malla nativa O1280, unos 9 km · datos de Open-Meteo',fontsize=11,color='#66583d')
fig.text(.065,.879,f'Consulta: {now:%d/%m/%Y %H:%M} UTC   |   Periodo: {start:%d/%m %H}–{xt[-1]:%d/%m %H} UTC   |   Pasada no indicada por la API',fontsize=10,color='#66583d')
fig.text(.065,.858,'ACUERDO ENTRE LOS ESCENARIOS · por variable y día',fontsize=15,weight='bold',color='#473b2a')
fig.text(.065,.841,'Mayor acuerdo = más confianza orientativa. Se muestra el bloque con más dudas de cada día; no es una garantía de acierto.',fontsize=10,color='#66583d')
ta=fig.add_axes([.065,.747,.90,.083]);ta.axis('off')
colnames=[d.strftime('%d/%m')+(' *' if d in [dates[0],dates[-1]] else '') for d in dates]
tab=ta.table(cellText=[[label]+row for label,row in zip(labels,table)],colLabels=['Variable']+colnames,cellLoc='center',colWidths=[.25]+[.75/len(dates)]*len(dates),bbox=[0,0,1,1])
tab.auto_set_font_size(False);tab.set_fontsize(10)
colors={'Mayor':'#d9ead3','Intermedio':'#fff0bf','Poco':'#f4d6ca','Sin datos':'#eeeeee'}
for (r,c),cell in tab.get_celld().items():
 cell.set_edgecolor('#d3c6a0');cell.set_linewidth(.5)
 if r==0:cell.set_facecolor('#ede2bd');cell.set_text_props(weight='bold',color='#473b2a')
 elif c==0:cell.set_facecolor('#fffdf6');cell.set_text_props(ha='left',color='#473b2a')
 else:cell.set_facecolor(colors[table[r-1][c-1]])
fig.text(.065,.736,'* Días parciales: solo se valora el tramo representado. Todas las fechas y horas son UTC.',fontsize=9,color='#66583d')
y=.721
for sentence in sentences:
 wrapped=textwrap.fill(sentence,170);fig.text(.065,y,wrapped,fontsize=9.5,color='#473b2a',va='top',linespacing=1.2);y-=.012*len(wrapped.splitlines())
notes=[
 'Cómo leerlo: la línea es la mediana; las bandas contienen el 50 % y el 80 % central de los escenarios. No son intervalos de confianza calibrados ni abarcan todas las posibilidades.',
 'Viento: velocidad a 10 m, no rachas. Dirección: de dónde sopla; ocho sectores de 45°, calma <1 km/h. El color indica el porcentaje de escenarios.',
 'Lluvia: acumulación por miembro antes de calcular percentiles. Las barras muestran el porcentaje de los 51 escenarios con al menos 0,2 mm en tres horas; no son probabilidades calibradas.',
 f'Fuente: ECMWF, a través de Open-Meteo · CC BY 4.0. Punto solicitado: 36,17 N / 5,35 O; punto devuelto: {raw['latitude']:.5f} N / {abs(raw['longitude']):.5f} O. Los datos horarios pueden estar interpolados temporalmente.',
 f'Cobertura: se representan solo horas con los 51 miembros completos, hasta {xt[-1]:%d/%m %H} UTC; no se mezcla con datos de 0,25°. Hora peninsular: UTC+1 en estas fechas.'
]
y=.077
for note in notes:
 wrapped=textwrap.fill(note,185);fig.text(.065,y,wrapped,fontsize=8.2,color='#66583d',va='top',linespacing=1.25);y-=.013
fig.savefig(ROOT/'meteograma.png',dpi=200,facecolor=fig.get_facecolor());plt.close(fig)
(ROOT/'Interpretacion.txt').write_text('ACUERDO POR DÍA (UTC)\n'+str(dict(zip(labels,table)))+'\n\n'+summary+'\n\n'+'\n\n'.join(notes)+'\n\nConsulta: '+query['url']+'\n')
print(json.dumps(statistics,ensure_ascii=False,indent=2))
