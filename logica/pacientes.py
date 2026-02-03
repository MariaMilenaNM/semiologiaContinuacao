pct1 = [None, "Sinto manchas roxas pelo corpo e sangramento na gengiva.", "Petéquias e equimoses em membros inferiores.", "Plaquetopenia severa (15.000/mm³).", None, ["Prescrever Corticoide e repouso."]]
pct2 = [None, "Sinto muito cansaço, formigamento nas mãos e esquecimento.", "Palidez cutâneo-mucosa e glossite (língua lisa).", "Anemia macrocítica e níveis baixos de B12.", None, ["Reposição de Vitamina B12 (Cianocobalamina)."]]
pct3 = [None, "Tenho movimentos involuntários e muita irritabilidade.", "Coreia (movimentos rápidos e desordenados).", "Presença de acantócitos no sangue periférico.", None, ["Suporte multidisciplinar e controle de sintomas motores."]]
pct4 = [None, "Minha urina sai muito escura pela manhã e sinto fraqueza.", "Icterícia leve (olhos amarelados).", "Teste de Ham positivo e LDH elevado.", None, ["Eculizumabe e suplementação de ácido fólico."]]
pct5 = [None, "Sinto muita fraqueza, palidez e tive várias infecções seguidas.", "Palidez intensa e ausência de linfonodomegalias.", "Pancitopenia e medula óssea hipocelular.", None, ["Transplante de Medula Óssea ou Terapia Imunossupressora."]]
pct6 = [None, "Febre persistente e sangramentos inexplicáveis.", "Equimoses espalhadas e febre de 38.5°C.", "Presença de bastonetes de Auer no mielograma.", None, ["Quimioterapia com ATRA (Ácido All-Trans Retinoico)."]]
pct7 = [None, "Meus pés formigam e notei manchas escuras na pele.", "Polineuropatia periférica e organomegalia.", "Pico monoclonal de IgA e lesões ósseas.", None, ["Tratamento da doença de base (mieloma/plasmocitoma)."]]
pct8 = [None, "Tenho suores noturnos e apareceram caroços no pescoço.", "Linfonodomegalia cervical indolor e firme.", "Biopsia de linfonodo positiva para Linfoma não Hodgkin.", None, ["Protocolo de quimioterapia (CHOP) e TARV para HIV."]]
pct9 = [None, "Comecei a ter calafrios e febre logo após a transfusão.", "Taquicardia e temperatura de 39°C.", "Cultura de sangue negativa, teste de Coombs negativo.", None, ["Antitérmicos e suspensão temporária da transfusão."]]
pct10 = [None, "Sinto muita falta de ar logo após receber o sangue.", "Crepitações pulmonares bilaterais e turgência jugular.", "BNP elevado e edema pulmonar no Raio-X.", None, ["Diuréticos (Furosemida) e suporte de oxigênio."]]


PACIENTES = [
    {
        "id": 1,
        "nome": "Arthur",
        "idade": 10,
        "registro": "071103",
        "respostas": pct1[1],
        "exame_fisico": pct1[2],
        "exames": pct1[3],
        "diagnosticos_possiveis": [
            "Púrpura Trombocitopênica Imune (PTI)",
            "Leucemia Linfoblástica Aguda",
            "Lúpus Eritematoso Sistêmico"
        ],
        "diagnostico_correto": "Púrpura Trombocitopênica Imune (PTI)",
        "plano": pct1[5][0],
        "ml": {
        "modelo": "modelos/H1/model.keras", # O treino salva como model.keras
        "words": "modelos/H1/words.pkl",
        "classes": "modelos/H1/classes.pkl",
        "intents": "intents/intentsH1.json"
}
    },
    {
        "id": 2,
        "nome": "José",
        "idade": 62,
        "registro": "061000",
        "respostas": pct2[1],
        "exame_fisico": pct2[2],
        "exames": pct2[3],
        "diagnosticos_possiveis": [
            "Anemia carencial por deficiência de B12, ácido fólico e ferro",
            "Anemia de doença crônica",
            "Deficiência de B12 secundária à Metformina"
        ],
        "diagnostico_correto": "Anemia carencial por deficiência de B12, ácido fólico e ferro",
        "plano": pct2[5][0],
        "ml": {
        "modelo": "modelos/H2/model.keras", # O treino salva como model.keras
        "words": "modelos/H2/words.pkl",
        "classes": "modelos/H2/classes.pkl",
        "intents": "intents/intentsH2.json"
}
    },
    {
        "id": 3,
        "nome": "Júlia",
        "idade": 24,
        "registro": "040807",
        "respostas": pct3[1],
        "exame_fisico": pct3[2],
        "exames": pct3[3],
        "diagnosticos_possiveis": [
            "Acantocitose Hereditária",
            "Doença de Huntington",
            "Doença de McLeod"
        ],
        "diagnostico_correto": "Acantocitose Hereditária",
        "plano": pct3[5][0],
        "ml": {
        "modelo": "modelos/H3/model.keras", # O treino salva como model.keras
        "words": "modelos/H3/words.pkl",
        "classes": "modelos/H3/classes.pkl",
        "intents": "intents/intentsH3.json"
}
    },
    {
        "id": 4,
        "nome": "Kleber",
        "idade": 32,
        "registro": "122269",
        "respostas": pct4[1],
        "exame_fisico": pct4[2],
        "exames": pct4[3],
        "diagnosticos_possiveis": [
            "Hemoglobinúria Paroxística Noturna",
            "Anemia Megaloblástica",
            "Neoplasia Hepatobiliar"
        ],
        "diagnostico_correto": "Hemoglobinúria Paroxística Noturna",
        "plano": pct4[5][0],
        "ml": {
        "modelo": "modelos/H4/model.keras", # O treino salva como model.keras
        "words": "modelos/H4/words.pkl",
        "classes": "modelos/H4/classes.pkl",
        "intents": "intents/intentsH4.json"
}
    },
    {
        "id": 5,
        "nome": "Jorge",
        "idade": 47,
        "registro": "081870",
        "respostas": pct5[1],
        "exame_fisico": pct5[2],
        "exames": pct5[3],
        "diagnosticos_possiveis": [
            "Anemia Aplásica Adquirida",
            "Síndrome Mielodisplásica",
            "Leucemia Mieloide Aguda"
        ],
        "diagnostico_correto": "Anemia Aplásica Adquirida",
        "plano": pct5[5][0],
        "ml": {
        "modelo": "modelos/H5/model.keras", # O treino salva como model.keras
        "words": "modelos/H5/words.pkl",
        "classes": "modelos/H5/classes.pkl",
        "intents": "intents/intentsH5.json"
}
    },
    {
        "id": 6,
        "nome": "Ícaro",
        "idade": 23,
        "registro": "072225",
        "respostas": pct6[1],
        "exame_fisico": pct6[2],
        "exames": pct6[3],
        "diagnosticos_possiveis": [
            "Leucemia Promielocítica Aguda",
            "Púrpura Trombocitopênica Imune",
            "Câncer de colo de útero"
        ],
        "diagnostico_correto": "Leucemia Promielocítica Aguda",
        "plano": pct6[5][0],
        "ml": {
        "modelo": "modelos/H6/model.keras", # O treino salva como model.keras
        "words": "modelos/H6/words.pkl",
        "classes": "modelos/H6/classes.pkl",
        "intents": "intents/intentsH6.json"
}
    },
    {
        "id": 7,
        "nome": "Antonio",
        "idade": 42,
        "registro": "122503",
        "respostas": pct7[1],
        "exame_fisico": pct7[2],
        "exames": pct7[3],
        "diagnosticos_possiveis": [
            "Síndrome POEMS",
            "Neuropatia Diabética",
            "Síndrome de Guillain-Barré"
        ],
        "diagnostico_correto": "Síndrome POEMS",
        "plano": pct7[5][0],
        "ml": {
        "modelo": "modelos/H7/model.keras", # O treino salva como model.keras
        "words": "modelos/H7/words.pkl",
        "classes": "modelos/H7/classes.pkl",
        "intents": "intents/intentsH7.json"
}
    },
    {
        "id": 8,
        "nome": "Mateus",
        "idade": 34,
        "registro": "051924",
        "respostas": pct8[1],
        "exame_fisico": pct8[2],
        "exames": pct8[3],
        "diagnosticos_possiveis": [
            "Linfoma não Hodgkin associado ao HIV",
            "Tuberculose ganglionar",
            "Linfadenopatia por infecção oportunista"
        ],
        "diagnostico_correto": "Linfoma não Hodgkin associado ao HIV",
        "plano": pct8[5][0],
        "ml": {
        "modelo": "modelos/H8/model.keras", # O treino salva como model.keras
        "words": "modelos/H8/words.pkl",
        "classes": "modelos/H8/classes.pkl",
        "intents": "intents/intentsH8.json"
        }
    },
    {
        "id": 9,
        "nome": "Marta",
        "idade": 45,
        "registro": "051540",
        "respostas": pct9[1],
        "exame_fisico": pct9[2],
        "exames": pct9[3],
        "diagnosticos_possiveis": [
            "Reação Febril Não Hemolítica",
            "Reação Hemolítica Aguda",
            "Contaminação Bacteriana"
        ],
        "diagnostico_correto": "Reação Febril Não Hemolítica",
        "plano": pct9[5][0],
        "ml": {
        "modelo": "modelos/H9/model.keras", # O treino salva como model.keras
        "words": "modelos/H9/words.pkl",
        "classes": "modelos/H9/classes.pkl",
        "intents": "intents/intentsH9.json"
}
    },
    {
        "id": 10,
        "nome": "João",
        "idade": 75,
        "registro": "082025",
        "respostas": pct10[1],
        "exame_fisico": pct10[2],
        "exames": pct10[3],
        "diagnosticos_possiveis": [
            "Sobrecarga Circulatória (TACO)",
            "TRALI",
            "Reação Alérgica Grave"
        ],
        "diagnostico_correto": "Sobrecarga Circulatória (TACO)",
        "plano": pct10[5][0],
        "ml": {
        "modelo": "modelos/H10/model.keras", # O treino salva como model.keras
        "words": "modelos/H10/words.pkl",
        "classes": "modelos/H10/classes.pkl",
        "intents": "intents/intentsH10.json"
}
    }
]
