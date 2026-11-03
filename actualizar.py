"""Descarga datos reales y genera una publicación nueva sin sobrescribir la anterior si falla."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,urllib.request,urllib.parse,tempfile,subprocess,sys,os,shutil,time
ROOT=Path(__file__).resolve().parent

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--datos-locales',type=Path,help='Reproducir una consulta guardada, sin descargar')
 args=parser.parse_args()
 with tempfile.TemporaryDirectory(prefix='meteograma-') as tmp:
  work=Path(tmp)
  if args.datos_locales:
   for name in ['respuesta.json','consulta.json']:shutil.copyfile(args.datos_locales/name,work/name)
  else:
   params={'latitude':36.17,'longitude':-5.35,'hourly':'temperature_2m,precipitation,wind_speed_10m,wind_direction_10m','models':'ecmwf_ifs_europe_ensemble','forecast_days':10,'timezone':'UTC','wind_speed_unit':'kmh'}
   url='https://ensemble-api.open-meteo.com/v1/ensemble?'+urllib.parse.urlencode(params)
   for attempt in range(3):
    try:
     with urllib.request.urlopen(url,timeout=90) as response:payload=response.read()
     data=json.loads(payload)
     if data.get('error') or not data.get('hourly'):raise ValueError('Respuesta de datos no válida')
     break
    except Exception:
     if attempt==2:raise
     time.sleep(3*(attempt+1))
   (work/'respuesta.json').write_bytes(payload)
   (work/'consulta.json').write_text(json.dumps({'url':url,'consulted_at_utc':datetime.now(timezone.utc).isoformat(),'model':params['models']},indent=2))
  subprocess.run([sys.executable,str(ROOT/'generar.py')],env=dict(os.environ,METEO_WORK=str(work)),check=True)
  out=ROOT/'out'
  if out.exists():shutil.rmtree(out)
  shutil.copytree(ROOT/'web',out)
  for path in work.iterdir():
   if path.is_file():shutil.copyfile(path,out/path.name)
  print('Publicación generada:',out)
if __name__=='__main__':main()
