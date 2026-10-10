import java.io.File;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.Set;
import org.semanticweb.elk.owlapi.ElkReasonerFactory;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.io.FileDocumentSource;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.reasoner.OWLReasoner;
import org.semanticweb.owlapi.search.EntitySearcher;

/**
 * Independent parse of one ontology file with the OWL API; prints one JSON line:
 * declared classes, SKOS concepts (class assertions to skos:Concept), named subClassOf axioms,
 * rdfs:label / skos labels, logical axiom count, and optionally ELK consistency + unsatisfiable classes.
 * Usage: java -Xmx8g -jar owl-verify.jar <file> [--elk]
 */
public class OwlVerify {
    static final String SKOS = "http://www.w3.org/2004/02/skos/core#";

    public static void main(String[] args) throws Exception {
        PrintStream out = new PrintStream(System.out, true, StandardCharsets.UTF_8.name());
        if (args[0].equals("--coherence")) {
            coherence(out, args[1], args[2], args[3]);
            return;
        }
        String path = args[0];
        boolean elk = args.length > 1 && args[1].equals("--elk");
        long start = System.currentTimeMillis();
        try {
            OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
            OWLOntologyLoaderConfiguration config = new OWLOntologyLoaderConfiguration()
                .setMissingImportHandlingStrategy(MissingImportHandlingStrategy.SILENT)
                .setStrict(false);
            OWLOntology o = manager.loadOntologyFromOntologyDocument(new FileDocumentSource(new File(path)), config);
            Set<OWLClass> declared = new HashSet<>();
            long objectProperties = 0, dataProperties = 0;
            for (OWLDeclarationAxiom ax : o.getAxioms(AxiomType.DECLARATION)) {
                if (ax.getEntity().isOWLClass()) declared.add(ax.getEntity().asOWLClass());
                else if (ax.getEntity().isOWLObjectProperty()) objectProperties++;
                else if (ax.getEntity().isOWLDataProperty()) dataProperties++;
            }
            long skosConcepts = o.getAxioms(AxiomType.CLASS_ASSERTION).stream()
                .filter(a -> !a.getClassExpression().isAnonymous()
                        && a.getClassExpression().asOWLClass().getIRI().toString().equals(SKOS + "Concept"))
                .count();
            long subclassNamed = o.getAxioms(AxiomType.SUBCLASS_OF).stream()
                .filter(a -> !a.getSubClass().isAnonymous() && !a.getSuperClass().isAnonymous()).count();
            long labels = 0;
            for (OWLAnnotationAssertionAxiom a : o.getAxioms(AxiomType.ANNOTATION_ASSERTION)) {
                String p = a.getProperty().getIRI().toString();
                if (p.endsWith("#label") || p.equals(SKOS + "prefLabel") || p.equals(SKOS + "altLabel")) labels++;
            }
            String consistent = "\"not_run\"";
            long unsat = -1;
            if (elk) {
                OWLReasoner r = new ElkReasonerFactory().createReasoner(o);
                boolean isConsistent = r.isConsistent();
                consistent = isConsistent ? "true" : "false";
                // An inconsistent ontology has no meaningful unsatisfiable-class set (every class is unsatisfiable);
                // report -1 and keep the parse counts instead of failing the whole file.
                if (isConsistent) unsat = r.getUnsatisfiableClasses().getEntitiesMinusBottom().size();
                r.dispose();
            }
            out.printf("{\"path\": %s, \"ok\": true, \"format\": %s, \"classes_declared\": %d, \"classes_signature\": %d, "
                + "\"skos_concepts\": %d, \"object_properties\": %d, \"data_properties\": %d, \"individuals\": %d, \"subclass_named\": %d, \"label_assertions\": %d, "
                + "\"logical_axioms\": %d, \"consistent_elk\": %s, \"unsatisfiable_elk\": %d, \"seconds\": %.1f}%n",
                q(path), q(String.valueOf(manager.getOntologyFormat(o))), declared.size(), o.getClassesInSignature().size(),
                skosConcepts, objectProperties, dataProperties, o.getIndividualsInSignature().size(), subclassNamed, labels, o.getLogicalAxiomCount(),
                consistent, unsat, (System.currentTimeMillis() - start) / 1000.0);
        } catch (Throwable t) {
            out.printf("{\"path\": %s, \"ok\": false, \"error\": %s}%n", q(path), q(t.getClass().getSimpleName() + ": " + String.valueOf(t.getMessage())));
        }
    }

    /**
     * Alignment coherence (OAEI "Coherent" column): merge source + target, add the mappings as axioms
     * (= -> equivalence, < -> subsumption, > -> reverse subsumption; classes or properties), classify with ELK,
     * and compare unsatisfiable classes before and after. mappings.tsv: source_iri, target_iri, relation.
     */
    static void coherence(PrintStream out, String src, String tgt, String mappings) throws Exception {
        long start = System.currentTimeMillis();
        OWLOntologyManager m = OWLManager.createOWLOntologyManager();
        OWLOntologyLoaderConfiguration config = new OWLOntologyLoaderConfiguration()
            .setMissingImportHandlingStrategy(MissingImportHandlingStrategy.SILENT).setStrict(false);
        OWLOntology a = m.loadOntologyFromOntologyDocument(new FileDocumentSource(new File(src)), config);
        OWLOntology b = m.loadOntologyFromOntologyDocument(new FileDocumentSource(new File(tgt)), config);
        OWLOntology merged = m.createOntology(IRI.create("urn:oa-rlaif:merged"));
        m.addAxioms(merged, a.getAxioms());
        m.addAxioms(merged, b.getAxioms());
        int before = unsat(merged);
        OWLDataFactory f = m.getOWLDataFactory();
        int added = 0;
        for (String line : java.nio.file.Files.readAllLines(new File(mappings).toPath(), StandardCharsets.UTF_8)) {
            String[] cols = line.split("\t");
            if (cols.length < 3 || cols[0].equals("source_iri")) continue;
            IRI s = IRI.create(cols[0]), t = IRI.create(cols[1]);
            boolean props = merged.containsObjectPropertyInSignature(s) && merged.containsObjectPropertyInSignature(t);
            OWLAxiom ax;
            if (props) {
                OWLObjectProperty ps = f.getOWLObjectProperty(s), pt = f.getOWLObjectProperty(t);
                ax = cols[2].equals("=") ? f.getOWLEquivalentObjectPropertiesAxiom(ps, pt)
                   : cols[2].equals(">") ? f.getOWLSubObjectPropertyOfAxiom(pt, ps) : f.getOWLSubObjectPropertyOfAxiom(ps, pt);
            } else {
                OWLClass cs = f.getOWLClass(s), ct = f.getOWLClass(t);
                ax = cols[2].equals("=") ? f.getOWLEquivalentClassesAxiom(cs, ct)
                   : cols[2].equals(">") ? f.getOWLSubClassOfAxiom(ct, cs) : f.getOWLSubClassOfAxiom(cs, ct);
            }
            m.addAxiom(merged, ax);
            added++;
        }
        int after = unsat(merged);
        out.printf("{\"ok\": true, \"mappings\": %d, \"unsatisfiable_before\": %d, \"unsatisfiable_after\": %d, "
            + "\"coherent\": %s, \"seconds\": %.1f}%n", added, before, after, after <= before ? "true" : "false",
            (System.currentTimeMillis() - start) / 1000.0);
    }

    static int unsat(OWLOntology o) {
        OWLReasoner r = new ElkReasonerFactory().createReasoner(o);
        r.precomputeInferences();
        int n = r.isConsistent() ? r.getUnsatisfiableClasses().getEntitiesMinusBottom().size() : Integer.MAX_VALUE;
        r.dispose();
        return n;
    }

    static String q(String s) {
        if (s == null) return "null";
        StringBuilder b = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            if (c == '"' || c == '\\') b.append('\\').append(c);
            else if (c < 0x20) b.append(' ');
            else b.append(c);
        }
        return b.append('"').toString();
    }
}
