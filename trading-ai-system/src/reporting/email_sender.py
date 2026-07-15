import os,smtplib
from email.mime.text import MIMEText
def send(html,subject):
 to=os.getenv('EMAIL_TO'); user=os.getenv('GMAIL_USER'); password=os.getenv('GMAIL_APP_PASSWORD'); sender=os.getenv('EMAIL_FROM') or user
 if not (to and user and password):return {'sent':False,'reason':'email_credentials_or_recipient_missing'}
 msg=MIMEText(html,'html');msg['Subject']=subject;msg['From']=sender;msg['To']=to
 with smtplib.SMTP_SSL('smtp.gmail.com',465,timeout=30) as s:s.login(user,password);s.sendmail(sender,[to],msg.as_string())
 return {'sent':True,'reason':'sent'}
