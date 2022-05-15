# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class Communication(models.Model):
    id = models.PositiveBigIntegerField(blank=True, null=False, primary_key=True)
    message = models.TextField(blank=True, null=True)
    subject = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'communication'


class CommunicationHistory(models.Model):
    id = models.PositiveBigIntegerField(blank=True, null=False, primary_key=True)
    message = models.TextField(blank=True, null=True)
    subject = models.TextField(blank=True, null=True)
    tblid = models.PositiveBigIntegerField(blank=True, null=True)
    tbltimestamp = models.DateTimeField(db_column='tblTimeStamp', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'communication_history'


class Item(models.Model):
    id = models.PositiveBigIntegerField(unique=True, blank=True, null=False, primary_key=True)
    description = models.TextField(blank=True, null=True)
    subject = models.TextField(blank=True, null=True)
    initials = models.CharField(max_length=100, blank=True, null=True)
    username = models.CharField(max_length=100, blank=True, null=True)
    password = models.CharField(max_length=100, blank=True, null=True)
    firstname = models.CharField(max_length=100, blank=True, null=True)
    lastname = models.CharField(max_length=100, blank=True, null=True)
    email = models.CharField(max_length=100, blank=True, null=True)
    phone = models.CharField(max_length=100, blank=True, null=True)
    ipphone = models.CharField(max_length=100, blank=True, null=True)
    mobile = models.CharField(max_length=100, blank=True, null=True)
    pager = models.CharField(max_length=100, blank=True, null=True)
    fax = models.CharField(max_length=100, blank=True, null=True)
    title = models.CharField(max_length=100, blank=True, null=True)
    sessionid = models.CharField(max_length=100, blank=True, null=True)
    lastlogon = models.CharField(max_length=100, blank=True, null=True)
    userlanguage = models.CharField(max_length=100, blank=True, null=True)
    guid = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'item'


class ItemHistory(models.Model):
    id = models.PositiveBigIntegerField(blank=True, null=False, primary_key=True)
    description = models.TextField(blank=True, null=True)
    subject = models.TextField(blank=True, null=True)
    initials = models.CharField(max_length=100, blank=True, null=True)
    username = models.CharField(max_length=100, blank=True, null=True)
    password = models.CharField(max_length=100, blank=True, null=True)
    firstname = models.CharField(max_length=100, blank=True, null=True)
    lastname = models.CharField(max_length=100, blank=True, null=True)
    email = models.CharField(max_length=100, blank=True, null=True)
    phone = models.CharField(max_length=100, blank=True, null=True)
    ipphone = models.CharField(max_length=100, blank=True, null=True)
    mobile = models.CharField(max_length=100, blank=True, null=True)
    pager = models.CharField(max_length=100, blank=True, null=True)
    fax = models.CharField(max_length=100, blank=True, null=True)
    title = models.CharField(max_length=100, blank=True, null=True)
    sessionid = models.CharField(max_length=100, blank=True, null=True)
    lastlogon = models.CharField(max_length=100, blank=True, null=True)
    userlanguage = models.CharField(max_length=100, blank=True, null=True)
    tblid = models.PositiveBigIntegerField(blank=True, null=True)
    tbltimestamp = models.DateTimeField(db_column='tblTimeStamp', blank=True, null=True)  # Field name made lowercase.
    guid = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'item_history'


class Object(models.Model):
    id = models.PositiveBigIntegerField(blank=True, null=False, primary_key=True)
    externalid = models.PositiveBigIntegerField(db_column='externalId', blank=True, null=True)  # Field name made lowercase.
    name = models.CharField(max_length=100, blank=True, null=True)
    objecttype = models.CharField(db_column='objectType', max_length=100, blank=True, null=True)  # Field name made lowercase.
    createdate = models.DateTimeField(db_column='createDate', blank=True, null=True)  # Field name made lowercase.
    createdby = models.PositiveBigIntegerField(db_column='createdBy', blank=True, null=True)  # Field name made lowercase.
    altereddate = models.DateTimeField(db_column='alteredDate', blank=True, null=True)  # Field name made lowercase.
    alteredby = models.PositiveBigIntegerField(db_column='alteredBy', blank=True, null=True)  # Field name made lowercase.
    state = models.CharField(max_length=100, blank=True, null=True)
    metatype = models.CharField(db_column='metaType', max_length=100, blank=True, null=True)  # Field name made lowercase.
    syncid = models.CharField(max_length=100, blank=True, null=True)
    indicatorchangedate = models.DateTimeField(blank=True, null=True)
    enterpriceroot = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'object'


class ObjectHistory(models.Model):
    id = models.PositiveBigIntegerField(blank=True, null=False, primary_key=True)
    externalid = models.PositiveBigIntegerField(db_column='externalId', blank=True, null=True)  # Field name made lowercase.
    name = models.CharField(max_length=100, blank=True, null=True)
    objecttype = models.CharField(db_column='objectType', max_length=100, blank=True, null=True)  # Field name made lowercase.
    createdate = models.DateTimeField(db_column='createDate', blank=True, null=True)  # Field name made lowercase.
    createdby = models.PositiveBigIntegerField(db_column='createdBy', blank=True, null=True)  # Field name made lowercase.
    altereddate = models.DateTimeField(db_column='alteredDate', blank=True, null=True)  # Field name made lowercase.
    alteredby = models.PositiveBigIntegerField(db_column='alteredBy', blank=True, null=True)  # Field name made lowercase.
    state = models.CharField(max_length=100, blank=True, null=True)
    tblid = models.PositiveBigIntegerField(blank=True, null=True)
    owntimestamp = models.DateTimeField(db_column='ownTimeStamp', blank=True, null=True)  # Field name made lowercase.
    metatype = models.CharField(db_column='metaType', max_length=100, blank=True, null=True)  # Field name made lowercase.
    indicatorchangedate = models.DateTimeField(blank=True, null=True)
    enterpriceroot = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'object_history'


class RelationExportTasktype(models.Model):
    id = models.PositiveBigIntegerField(blank=True, null=False, primary_key=True)
    leftid = models.PositiveBigIntegerField(db_column='leftId', blank=True, null=True)  # Field name made lowercase.
    rightid = models.PositiveBigIntegerField(db_column='rightId', blank=True, null=True)  # Field name made lowercase.
    relationtypeid = models.IntegerField(db_column='relationTypeID', blank=True, null=True)  # Field name made lowercase.
    lefttype = models.CharField(db_column='leftType', max_length=100, blank=True, null=True)  # Field name made lowercase.
    righttype = models.CharField(db_column='rightType', max_length=100, blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'relation_export_tasktype'


class RelationHistory(models.Model):
    id = models.PositiveBigIntegerField(blank=True, null=False, primary_key=True)
    leftid = models.PositiveBigIntegerField(db_column='leftId', blank=True, null=True)  # Field name made lowercase.
    rightid = models.PositiveBigIntegerField(db_column='rightId', blank=True, null=True)  # Field name made lowercase.
    relationtypeid = models.IntegerField(db_column='relationTypeID', blank=True, null=True)  # Field name made lowercase.
    lefttype = models.CharField(db_column='leftType', max_length=100, blank=True, null=True)  # Field name made lowercase.
    righttype = models.CharField(db_column='rightType', max_length=100, blank=True, null=True)  # Field name made lowercase.
    tblid = models.IntegerField(blank=True, null=True)
    tbltimestamp = models.DateTimeField(db_column='tblTimeStamp', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'relation_history'


class Request(models.Model):
    id = models.PositiveBigIntegerField(unique=True, blank=True, null=False, primary_key=True)
    description = models.TextField(blank=True, null=True)
    itilstate = models.CharField(db_column='itilState', max_length=100, blank=True, null=True)  # Field name made lowercase.
    receiveddate = models.TextField(db_column='receivedDate', blank=True, null=True)  # Field name made lowercase.
    receivedvia = models.CharField(db_column='receivedVia', max_length=100, blank=True, null=True)  # Field name made lowercase.
    servicedisorderdate = models.TextField(db_column='serviceDisorderDate', blank=True, null=True)  # Field name made lowercase.
    automaticallycreated = models.CharField(db_column='automaticallyCreated', max_length=100, blank=True, null=True)  # Field name made lowercase.
    callbackmethod = models.CharField(db_column='callBackMethod', max_length=100, blank=True, null=True)  # Field name made lowercase.
    cost = models.CharField(max_length=100, blank=True, null=True)
    timeconsumption = models.CharField(db_column='timeConsumption', max_length=100, blank=True, null=True)  # Field name made lowercase.
    numberofalarms = models.CharField(db_column='numberOfAlarms', max_length=100, blank=True, null=True)  # Field name made lowercase.
    subject = models.CharField(max_length=100, blank=True, null=True)
    classification = models.CharField(max_length=100, blank=True, null=True)
    priority = models.CharField(max_length=100, blank=True, null=True)
    impact = models.CharField(max_length=100, blank=True, null=True)
    urgency = models.CharField(max_length=100, blank=True, null=True)
    deadline = models.TextField(blank=True, null=True)
    investigationanddiagnosis = models.CharField(db_column='investigationAndDiagnosis', max_length=100, blank=True, null=True)  # Field name made lowercase.
    cause = models.CharField(max_length=100, blank=True, null=True)
    workaround = models.CharField(db_column='workAround', max_length=100, blank=True, null=True)  # Field name made lowercase.
    solution = models.TextField(blank=True, null=True)
    solutiondate = models.TextField(db_column='solutionDate', blank=True, null=True)  # Field name made lowercase.
    closingcode = models.CharField(db_column='closingCode', max_length=100, blank=True, null=True)  # Field name made lowercase.
    expectedsolutiondatetime = models.TextField(db_column='expectedSolutionDateTime', blank=True, null=True)  # Field name made lowercase.
    impactifimplemented = models.CharField(db_column='impactIfImplemented', max_length=100, blank=True, null=True)  # Field name made lowercase.
    impactifnotimplemented = models.CharField(db_column='impactIfNotImplemented', max_length=100, blank=True, null=True)  # Field name made lowercase.
    expectedresourcecost = models.CharField(db_column='expectedResourceCost', max_length=100, blank=True, null=True)  # Field name made lowercase.
    approvalofcontentoftherfc = models.CharField(db_column='approvalOfContentOfTheRFC', max_length=100, blank=True, null=True)  # Field name made lowercase.
    starttime = models.TextField(blank=True, null=True)
    endtime = models.TextField(blank=True, null=True)
    currentsection = models.CharField(db_column='currentSection', max_length=100, blank=True, null=True)  # Field name made lowercase.
    previoussections = models.CharField(db_column='previousSections', max_length=100, blank=True, null=True)  # Field name made lowercase.
    permanentlycloseddate = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'request'


class RequestHistory(models.Model):
    id = models.PositiveBigIntegerField(blank=True, null=False, primary_key=True)
    description = models.TextField(blank=True, null=True)
    itilstate = models.CharField(db_column='itilState', max_length=100, blank=True, null=True)  # Field name made lowercase.
    receiveddate = models.DateTimeField(db_column='receivedDate', blank=True, null=True)  # Field name made lowercase.
    receivedvia = models.CharField(db_column='receivedVia', max_length=100, blank=True, null=True)  # Field name made lowercase.
    servicedisorderdate = models.DateTimeField(db_column='serviceDisorderDate', blank=True, null=True)  # Field name made lowercase.
    automaticallycreated = models.CharField(db_column='automaticallyCreated', max_length=100, blank=True, null=True)  # Field name made lowercase.
    callbackmethod = models.CharField(db_column='callBackMethod', max_length=100, blank=True, null=True)  # Field name made lowercase.
    cost = models.CharField(max_length=100, blank=True, null=True)
    timeconsumption = models.CharField(db_column='timeConsumption', max_length=100, blank=True, null=True)  # Field name made lowercase.
    numberofalarms = models.CharField(db_column='numberOfAlarms', max_length=100, blank=True, null=True)  # Field name made lowercase.
    subject = models.CharField(max_length=100, blank=True, null=True)
    classification = models.CharField(max_length=100, blank=True, null=True)
    priority = models.CharField(max_length=100, blank=True, null=True)
    impact = models.CharField(max_length=100, blank=True, null=True)
    urgency = models.CharField(max_length=100, blank=True, null=True)
    deadline = models.DateTimeField(blank=True, null=True)
    investigationanddiagnosis = models.CharField(db_column='investigationAndDiagnosis', max_length=100, blank=True, null=True)  # Field name made lowercase.
    cause = models.CharField(max_length=100, blank=True, null=True)
    workaround = models.CharField(db_column='workAround', max_length=100, blank=True, null=True)  # Field name made lowercase.
    solution = models.TextField(blank=True, null=True)
    solutiondate = models.DateTimeField(db_column='solutionDate', blank=True, null=True)  # Field name made lowercase.
    closingcode = models.CharField(db_column='closingCode', max_length=100, blank=True, null=True)  # Field name made lowercase.
    expectedsolutiondatetime = models.DateTimeField(db_column='expectedSolutionDateTime', blank=True, null=True)  # Field name made lowercase.
    impactifimplemented = models.CharField(db_column='impactIfImplemented', max_length=100, blank=True, null=True)  # Field name made lowercase.
    impactifnotimplemented = models.CharField(db_column='impactIfNotImplemented', max_length=100, blank=True, null=True)  # Field name made lowercase.
    expectedresourcecost = models.CharField(db_column='expectedResourceCost', max_length=100, blank=True, null=True)  # Field name made lowercase.
    approvalofcontentoftherfc = models.CharField(db_column='approvalOfContentOfTheRFC', max_length=100, blank=True, null=True)  # Field name made lowercase.
    starttime = models.DateTimeField(blank=True, null=True)
    endtime = models.DateTimeField(blank=True, null=True)
    currentsection = models.CharField(db_column='currentSection', max_length=100, blank=True, null=True)  # Field name made lowercase.
    tblid = models.PositiveBigIntegerField(blank=True, null=True)
    tbltimestamp = models.DateTimeField(db_column='tblTimeStamp', blank=True, null=True)  # Field name made lowercase.
    previoussections = models.CharField(db_column='previousSections', max_length=100, blank=True, null=True)  # Field name made lowercase.
    permanentlycloseddate = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'request_history'


class Relation(models.Model):
    id = models.PositiveBigIntegerField(blank=True, null=False, primary_key=True)
    leftid = models.PositiveBigIntegerField(db_column='leftId', blank=True, null=True)  # Field name made lowercase.
    rightid = models.PositiveBigIntegerField(db_column='rightId', blank=True, null=True)  # Field name made lowercase.
    relationtypeid = models.IntegerField(db_column='relationTypeID', blank=True, null=True)  # Field name made lowercase.
    lefttype = models.CharField(db_column='leftType', max_length=100, blank=True, null=True)  # Field name made lowercase.
    righttype = models.CharField(db_column='rightType', max_length=100, blank=True, null=True)  # Field name made lowercase.

    request = models.ForeignKey(Request, on_delete=models.CASCADE, db_column='leftid')

    class Meta:
        managed = False
        db_table = 'relation'
