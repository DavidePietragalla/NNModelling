Si vuole progettare una Domain Specific Language che permetta di modellare ed addestrare reti neurali utrilizzando un linguaggio diagrammatico.

Il seguente linguaggio è formato dai seguenti componenti:

1. Nodi
  1.1. Un nodo può essere un Modulo
    1.1.1. Un Modulo può ricevere solo una connessione, ma può connettersi a tanti altri nodi.
    1.1.2. Un Modulo è associato ad una expression language, ovvero il codice che viene eseguito quando il token passa attraverso al modulo.
  1.2. Un nodo può essere di input, quindi avrà solo una connessione verso l'esterno.
  1.3. Esistono anche nodi di output, che sono o le loss function oppure degli output non definiti. L'utente deve poter scegliere che tipo di output vuole dal modello.
  1.4. Un nuodo può essere un sottomodello.
    1.4.1. Un sottomodello può essere caricato da un modello esistente/
    1.4.2. L'utente può definire sottomodelli dentro un modello.
    1.4.3. Deve essere possibile salvare sul disco solo un sottomodello.
    1.4.4. Deve essere possibile avere sottomodelli dentro sottomodelli

2. Steriotipi
  2.1. Uno steriotipo si può applicare a qualsiasi modulo.
  2.2. Uno steriotipo può essere di diversi tipi:
    2.2.1. Steriotipo per definire l'expression di un Modulo. (Non si può applicare ai sottomodelli)
    2.2.2. Steriotipo per estendere il comportamente di un Modulo/Sottomodello.
      2.2.2.1. Esempi: prendi un modulo e mettilo in sequenza 100 volte. Prendi un modulo, moltiplicare 100 volte e passare l'input a 100 di questi moduli diversi, per poi fare un join alla fine.

3. Join
  3.1. I fork sono impliciti perchè da ogni nodo posso far partire quante connessioni io voglio.
  3.2. Per i join bisogna invece modellarli esplicitamente, poichè un modulo può accettare solo un input. Quindi un Join è un Nodo speciale che ha tanti input e un output.
  3.3. I Join però è associato un expression che esprime che tipo di Join è
  3.3.1. Esempi: Inner-production, Outer-product, somma, qualsiasi altra operazione tra tensori.
