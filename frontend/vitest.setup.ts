import "@testing-library/jest-dom/vitest";

// jsdom implements neither of these, and the evidence-card interaction depends
// on both. Stubbing them here lets the tests assert they were called.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = function scrollIntoView() {};
}
