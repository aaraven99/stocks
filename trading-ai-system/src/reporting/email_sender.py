"""Gmail SMTP delivery with schedule guard and an inline equity-curve image."""
import os,smtplib
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from zoneinfo import ZoneInfo


def _truthy(name):return os.getenv(name,'false').lower() in ('1','true','yes')


def send(html,subject):
 now=datetime.now(ZoneInfo('America/Chicago'));scheduled=_truthy('EMAIL_SCHEDULED_RUN');manual=_truthy('EMAIL_ALLOW_OFF_SCHEDULE')
 if not manual and not scheduled and (now.weekday()>4 or now.hour!=7):return {'sent':False,'reason':'outside_07_00_america_chicago_schedule','local_time':now.isoformat()}
 recipients=[item.strip() for item in os.getenv('EMAIL_TO','').split(',') if item.strip()];user=os.getenv('GMAIL_USER');password=os.getenv('GMAIL_APP_PASSWORD');sender=os.getenv('EMAIL_FROM') or user
 if not (recipients and user and password):return {'sent':False,'reason':'email_credentials_or_recipient_missing'}
 msg=MIMEMultipart('related');msg['Subject']=subject;msg['From']=sender;msg['To']=', '.join(recipients);alternative=MIMEMultipart('alternative');alternative.attach(MIMEText('Systematic Swing Research briefing. View the HTML version for tables and charts.','plain','utf-8'));alternative.attach(MIMEText(html,'html','utf-8'));msg.attach(alternative)
 chart=Path('artifacts/equity_curve.png')
 if chart.exists():
  image=MIMEImage(chart.read_bytes(),_subtype='png');image.add_header('Content-ID','<equity_curve>');image.add_header('Content-Disposition','inline',filename=chart.name);msg.attach(image)
 try:
  with smtplib.SMTP_SSL('smtp.gmail.com',465,timeout=30) as smtp:smtp.login(user,password);smtp.sendmail(sender,recipients,msg.as_string())
 except (OSError,smtplib.SMTPException) as exc:return {'sent':False,'reason':'smtp_failure','error_type':type(exc).__name__}
 return {'sent':True,'reason':'sent','local_time':now.isoformat(),'scheduled_run':scheduled,'recipients':len(recipients)}
