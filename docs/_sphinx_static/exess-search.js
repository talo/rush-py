(function () {
  "use strict";

  function shouldFilter() {
    var params = new URLSearchParams(window.location.search || "");
    var flag = params.get("exess");
    if (!flag) {
      return false;
    }
    return flag === "1" || flag.toLowerCase() === "true" || flag.toLowerCase() === "yes";
  }

  function applyFilter() {
    if (!shouldFilter()) {
      return;
    }

    if (!window.Search || typeof Search._performSearch !== "function") {
      window.setTimeout(applyFilter, 50);
      return;
    }

    if (Search._exessFiltered) {
      return;
    }
    Search._exessFiltered = true;

    var originalPerform = Search._performSearch;
    Search._performSearch = function (query, searchTerms, excludedTerms, highlightTerms, objectTerms) {
      var results = originalPerform.call(
        Search,
        query,
        searchTerms,
        excludedTerms,
        highlightTerms,
        objectTerms
      );
      return results.filter(function (item) {
        return typeof item[0] === "string" && item[0].indexOf("exess/") === 0;
      });
    };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", applyFilter);
  } else {
    applyFilter();
  }
})();
