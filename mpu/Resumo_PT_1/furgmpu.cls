% Fork de https://github.com/jjonw/furgmpu
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{furgmpu}[2023/06/05 Classe para trabalhos da MPU]

% Tipo artigo com fonte em 12pt e sem alternância de lado
\LoadClass[12pt,oneside]{article}

% Define tamanho A4, margens e posicionamentos de cabeçalho e rodapé
\RequirePackage[
	 paper=a4paper
	,top=2.5cm
	,right=2cm
	,bottom=4cm
	,left=3cm
	,headheight=2cm
	,headsep=0.4cm
	,footskip=0.5cm
]{geometry}

% Espaçamento antes de cabeçalho?
\addtolength{\voffset}{0.9cm}
% Codificação do conteúdo é em UTF-8
\RequirePackage[utf8]{inputenc}
% Tipo de fonte de 8 bits
\RequirePackage[T1]{fontenc}
% Cálculo de páginas
\RequirePackage{lastpage}
% Fonte semelhante à Helvetica
\RequirePackage{helvet}
% Seta fonte padrão a sem pé
\renewcommand{\familydefault}{\sfdefault}
% ?
\RequirePackage{etoolbox}
\patchcmd{\thebibliography}{\section*}{\section}{}{}
% Referências no padrão ABNT
\RequirePackage[alf,abnt-emphasize=bf]{abntex2cite}
% Configura alguns espaçamentos?
\RequirePackage{booktabs}
\RequirePackage{titling}
\pretitle{\center\vspace{-1.4cm}}
\preauthor{\flushright}
\postauthor{\vskip -1.0cm}
\posttitle{\vskip 0.4cm}
% Inclusão de imagens
\RequirePackage{graphicx}
% Formatação de seções?
\RequirePackage{sectsty}
\sectionfont{\fontsize{12}{12}\selectfont}
% Formatação de seções?
\RequirePackage{titlesec}
\titlespacing{\section}{0pt}{24pt}{0pt}
% Identação de primeira linha
\setlength{\parindent}{1.25cm}
% Gestão de cores
\RequirePackage{xcolor}
\definecolor{mpularanja}{RGB}{237,135,62}
\definecolor{mpulilas}{RGB}{84,47,90}
% Desenhos vetoriais
\RequirePackage{tikz}
% Cabeçalho e rodapé
\RequirePackage{fancyhdr}
% Mesclar linhas
\RequirePackage{multirow}
% Configura cabeçalho e rodapé
\fancypagestyle{plain}{
	% Reta configurações
	\fancyhf{}
	% Linhas separando do conteúdo com tamanho 0
	\renewcommand{\headrulewidth}{0pt}
	\renewcommand{\footrulewidth}{0pt}
	% Parte do meio do cabeçalho
	\fancyhead[C]{
		% Tabela
		\begin{tabular}{ c p{2.5mm} c }
			% Célula até a 4ª linha
			\multirow{4}{*}{
			\includegraphics[width=4.45cm]{header.png}
			}
			% Restante do cabeçalho
			&&\footnotesize Universidade Federal do Rio Grande – FURG\\
			% Editar edição
			&&\footnotesize 23ª Mostra da Produção Universitária – MPU\\
			&&\footnotesize Rio Grande/RS, Brasil, 25 a 27 de outubro de 2023\\
			&&\footnotesize ISSN: 2317-4420                                  
		\end{tabular}
	}
	% Número em rodapé
	\fancyfoot[R]{\thepage}
}

% Tipo de página
\pagestyle{plain}

% Comandos de título e de autores
\newcommand{\titulo}[1]{\title{\textbf{#1}}}
\newcommand{\autores}[4]{\author{{\normalsize{#1\\\vspace{0.5cm}#2\\#3\\#4}}}\date{\vspace{-4.2ex}}}
