import random

nombreHombre =["ANTONIO",
  "MANUEL",
  "JOSE",
  "FRANCISCO",
  "DAVID",
  "JUAN",
  "JAVIER",
  "JOSE ANTONIO",
  "DANIEL",
  "JOSE LUIS",
  "FRANCISCO JAVIER",
  "CARLOS",
  "JESUS",
  "ALEJANDRO",
  "MIGUEL",
  "JOSE MANUEL",
  "RAFAEL",
  "MIGUEL ANGEL",
  "PABLO",
  "PEDRO",
  "ANGEL",
  "SERGIO",
  "JOSE MARIA",
  "FERNANDO",
  "JORGE",
  "LUIS",
  "ALBERTO",
  "ALVARO",
  "JUAN CARLOS",
  "ADRIAN",
  "DIEGO",
  "JUAN JOSE",
  "RAUL",
  "IVAN",
  "JUAN ANTONIO",
  "RUBEN",
  "ENRIQUE",
  "OSCAR",
  "RAMON",
  "ANDRES",
  "VICENTE",
  "JUAN MANUEL",
  "SANTIAGO",
  "JOAQUIN",
  "VICTOR",
  "MARIO",
  "EDUARDO",
  "ROBERTO",
  "JAIME",
  "FRANCISCO JOSE",
  "MARCOS",
  "IGNACIO",
  "HUGO",
  "ALFONSO",
  "JORDI",
  "RICARDO",
  "SALVADOR",
  "GUILLERMO",
  "GABRIEL",
  "MARC",
  "EMILIO",
  "MOHAMED",
  "GONZALO",
  "JULIO",
  "JULIAN",
  "MARTIN",
  "JOSE MIGUEL",
  "TOMAS",
  "AGUSTIN",
  "NICOLAS",
  "JOSE RAMON",
  "SAMUEL",
  "ISMAEL",
  "JOAN",
  "CRISTIAN",
  "FELIX",
  "LUCAS",
  "AITOR",
  "HECTOR",
  "JUAN FRANCISCO",
  "IKER",
  "ALEX",
  "JOSE CARLOS",
  "JOSEP",
  "SEBASTIAN",
  "MARIANO",
  "CESAR",
  "ALFREDO",
  "DOMINGO",
  "JOSE ANGEL",
  "FELIPE",
  "VICTOR MANUEL",
  "RODRIGO",
  "JOSE IGNACIO",
  "MATEO",
  "LUIS MIGUEL",
  "JOSE FRANCISCO",
  "JUAN LUIS",
  "XAVIER",
  "ALBERT"]

nombreMujer = ["MARIA CARMEN",
  "MARIA",
  "CARMEN",
  "ANA MARIA",
  "JOSEFA",
  "MARIA PILAR",
  "ISABEL",
  "LAURA",
  "MARIA DOLORES",
  "MARIA TERESA",
  "ANA",
  "CRISTINA",
  "MARTA",
  "MARIA ANGELES",
  "LUCIA",
  "FRANCISCA",
  "MARIA ISABEL",
  "MARIA JOSE",
  "ANTONIA",
  "DOLORES",
  "SARA",
  "PAULA",
  "ELENA",
  "MARIA LUISA",
  "RAQUEL",
  "ROSA MARIA",
  "PILAR",
  "MANUELA",
  "CONCEPCION",
  "MARIA JESUS",
  "MERCEDES",
  "JULIA",
  "BEATRIZ",
  "NURIA",
  "SILVIA",
  "ALBA",
  "IRENE",
  "ROSARIO",
  "JUANA",
  "TERESA",
  "PATRICIA",
  "ENCARNACION",
  "MONTSERRAT",
  "ANDREA",
  "ROCIO",
  "MONICA",
  "ALICIA",
  "MARIA MAR",
  "ROSA",
  "SONIA",
  "SANDRA",
  "MARINA",
  "ANGELA",
  "SUSANA",
  "NATALIA",
  "YOLANDA",
  "MARGARITA",
  "MARIA JOSEFA",
  "CLAUDIA",
  "SOFIA",
  "EVA",
  "CARLA",
  "MARIA ROSARIO",
  "INMACULADA",
  "MARIA MERCEDES",
  "ANA ISABEL",
  "ESTHER",
  "NOELIA",
  "VERONICA",
  "NEREA",
  "CAROLINA",
  "ANGELES",
  "DANIELA",
  "MARIA VICTORIA",
  "EVA MARIA",
  "INES",
  "MIRIAM",
  "LORENA",
  "MARIA ROSA",
  "MARIA ELENA",
  "ANA BELEN",
  "VICTORIA",
  "MARIA CONCEPCION",
  "AMPARO",
  "MARTINA",
  "MARIA ANTONIA",
  "ALEJANDRA",
  "LIDIA",
  "CATALINA",
  "CELIA",
  "MARIA NIEVES",
  "CONSUELO",
  "FATIMA",
  "OLGA",
  "AINHOA",
  "GLORIA",
  "CLARA",
  "MARIA CRISTINA",
  "MARIA SOLEDAD",
  "EMILIA"]

apellidos = ["MARTINEZ",
  "GARCIA",
  "FERNANDEZ",
  "PEREZ",
  "JIMENEZ",
  "GONZALEZ",
  "RUIZ",
  "LOPEZ",
  "SAENZ",
  "RODRIGUEZ",
  "GOMEZ",
  "MORENO",
  "HERNANDEZ",
  "SANCHEZ",
  "ALONSO",
  "PASCUAL",
  "GIL",
  "MARIN",
  "DIEZ",
  "ALVAREZ",
  "GUTIERREZ",
  "MARTIN",
  "CALVO",
  "BLANCO",
  "EZQUERRO",
  "RUBIO",
  "IBAÑEZ",
  "MUÑOZ",
  "GARRIDO",
  "SAEZ",
  "DIAZ",
  "RAMIREZ",
  "PALACIOS",
  "ORTEGA",
  "BENITO",
  "SANTAMARIA",
  "ROMERO",
  "OCHOA",
  "LEON",
  "DOMINGUEZ",
  "HERCE",
  "PEÑA",
  "GABARRI",
  "MERINO",
  "TORRES",
  "SAINZ",
  "SANZ",
  "CASTILLO",
  "ORTIZ",
  "CORDON"]  

grado = ['1º', '2º', '3º', '4º', '5º', '6º', '7º', '8º', '9º', '10º', '11º']
estado = ['Ganado', 'Perdido', 'Desertado']
departamento_aux=["Amazonas",
"Antioquia",
"Arauca",
"Atlántico",
"Bogotá",
"Bolívar",
"Boyacá",
"Caldas",
"Caquetá",  
"Casanare",
"Cauca",
"Cesar",
"Chocó",
"Córdoba",
"Cundinamarca",
"Guainía",  
"Guaviare",
"Huila",
"La Guajira",
"Magdalena",
"Meta",
"Nariño",
"Norte de Santander",
"Putumayo",
"Quindío",
"Risaralda",
"San Andrés y Providencia",
"Santander",
"Sucre",
"Tolima",
"Valle del Cauca",
"Vaupés",
"Vichada"]

edad_lista=[]
departamento_lista=[]
lista_estado=[]

def generarNombreSexoEdad():
  sexo = random.choice(['Masculino','Femenino'])
  if sexo =='Masculino':
    nombre = random.choice(nombreHombre)
  else:
    nombre = random.choice(nombreMujer)
  edad = 4
  apellido1 = random.choice(apellidos)
  apellido2 = random.choice(apellidos)
  ano = random.randint(1950, 2022)
  ano_inicial=ano
  departamento=random.choice(departamento_aux)

  dic = {}
  estado = ['Aprobado', 'Aprobado' ,'Perdido', 'Aprobado', 'Desertado', 'Aprobado',"Aprobado","Perdido","Aprobado","Aprobado","Abandono"]
  contador_aprobado=0
  contador_perdido=0
  contador_abandono=0
  contador_desertado=0

  for i in grado:
    estadoaux = random.choice(estado)
    if estadoaux == 'Aprobado':
      dic.update({ano:{i:estadoaux}})
      contador_aprobado+=1
    elif estadoaux == 'Perdido':
      dic.update({ano:{i:estadoaux}})
      ano+=1
      contador_perdido+=1
      dic.update({ano:{i:'Aprobado'}})
    elif estadoaux=="Abandono":
      dic.update({ano:{i:estadoaux}})
      break
    else:
      dic.update({ano:{i:estadoaux}})
      ano = random.randint(ano+1, ano+5)
      contador_desertado+=1
    ano += 1
  edad = ano-ano_inicial+5

  #lista edad
  global edad_lista
  edad_lista=[edad]+edad_lista

  #lista depto
  global departamento_lista
  departamento_lista=[departamento]+departamento_lista

  #lista estados
  global lista_estado
  lista_estado=[estado]+lista_estado

  return {'Nombre':nombre+" "+apellido1+" "+apellido2,
            'Sexo':sexo,
            'Edad':edad,
            'Cursos': dic,
            "Departamento":departamento,
          }
  
