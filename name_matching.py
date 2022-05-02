# /******************
#  * Name Matching
#  *
#  *   At Checkr, one of the most important aspects of our work is accurately matching records
#  * to candidates. One of the ways that we do this is by comparing the name on a given record
#  * to a list of known aliases for the candidate. In this exercise, we will implement a
#  * `nameMatch` method that accepts the list of known aliases as well as the name returned
#  * on a record. It should return true if the name matches any of the aliases and false otherwise.
#  *
#  * The nameMatch method will be required to pass the following tests:
#  *
#  * 1. Exact match
#  *
#  *   knownAliases = ["Alphonse Gabriel Capone", "Al Capone"]
#  *   nameMatch(knownAliases, "Alphonse Gabriel Capone") => true
#  *   nameMatch(knownAliases, "Al Capone")               => true
#  *   nameMatch(knownAliases, "Alphonse Francis Capone") => false
#  *
#  *
#  * 2. Middle name missing (on alias)
#  *
#  *   knownAliases = ["Alphonse Capone"]
#  *   nameMatch(knownAliases, "Alphonse Gabriel Capone") => true
#  *   nameMatch(knownAliases, "Alphonse Francis Capone") => true
#  *   nameMatch(knownAliases, "Alexander Capone")        => false
#  *
#  *
#  * 3. Middle name missing (on record name)
#  *
#  *   knownAliases = ["Alphonse Gabriel Capone"]
#  *   nameMatch(knownAliases, "Alphonse Capone")         => true
#  *   nameMatch(knownAliases, "Alphonse Francis Capone") => false
#  *   nameMatch(knownAliases, "Alexander Capone")        => false
#  *
#  *
#  * 4. More middle name tests
#  *    These serve as a sanity check of your implementation of cases 2 and 3
#  *
#  *   knownAliases = ["Alphonse Gabriel Capone", "Alphonse Francis Capone"]
#  *   nameMatch(knownAliases, "Alphonse Gabriel Capone") => true
#  *   nameMatch(knownAliases, "Alphonse Francis Capone") => true
#  *   nameMatch(knownAliases, "Alphonse Edward Capone")  => false
#  *
#  *
#  * 5. Middle initial matches middle name
#  *
#  *   knownAliases = ["Alphonse Gabriel Capone", "Alphonse F Capone"]
#  *   nameMatch(knownAliases, "Alphonse G Capone")       => true
#  *   nameMatch(knownAliases, "Alphonse Francis Capone") => true
#  *   nameMatch(knownAliases, "Alphonse E Capone")       => false
#  *   nameMatch(knownAliases, "Alphonse Edward Capone")  => false
#  *   nameMatch(knownAliases, "Alphonse Gregory Capone") => false
#  *
#  *
#  * Bonus: Transposition
#  *
#  * Transposition (swapping) of the first name and middle name is relatively common.
#  * In order to accurately match the name returned from a record we should take this
#  * into account.
#  *
#  * All of the test cases implemented previously also apply to the transposed name.
#  *
#  *
#  * 6. First name and middle name can be transposed
#  *
#  *   "Gabriel Alphonse Capone" is a valid transposition of "Alphonse Gabriel Capone"
#  *
#  *   knownAliases = ["Alphonse Gabriel Capone"]
#  *   nameMatch(knownAliases, "Gabriel Alphonse Capone") => true
#  *   nameMatch(knownAliases, "Gabriel A Capone")        => true
#  *   nameMatch(knownAliases, "Gabriel Capone")          => true
#  *   nameMatch(knownAliases, "Gabriel Francis Capone")  => false
#  *
#  *
#  * 7. Last name cannot be transposed
#  *
#  *   "Alphonse Capone Gabriel" is NOT a valid transposition of "Alphonse Gabriel Capone"
#  *   "Capone Alphonse Gabriel" is NOT a valid transposition of "Alphonse Gabriel Capone"
#  *
#  *   knownAliases = ["Alphonse Gabriel Capone"]
#  *   nameMatch(knownAliases, "Alphonse Capone Gabriel") => false
#  *   nameMatch(knownAliases, "Capone Alphonse Gabriel") => false
#  *   nameMatch(knownAliases, "Capone Gabriel")          => false
#  */

# // function hasMiddleName(name) {
# //   let splitName = name.split(' ');
  
# //   return splitName.length === 3;
# // }

# // function nameWithoutMiddleName(name) {
# //   if (hasMiddleName(name) {
# //     return [splitName[0], splitName[2]].join(' ');
# //   } 

# //   return name
# // }


# // function nameWithMiddleInitial(name) {  
# //   if (hasMiddleName(name) {
# //     return [splitName[0], splitName[1][0], splitName[2]].join(' ');
# //   } else {
# //     return name
# //   }
# // }

# // function nameMatch(knownAliases, name) {
# //   // Implement me
# //   let hasExactMatch = knownAliases.includes(name);
  
# //   if (hasExactMatch) {
# //     return hasExactMatch;
# //   }
  
# //   // Find a match with the middle name missing on the provided name
# //   let hasFuzzyMiddleNameMatch = knownAliases.includes(nameWithoutMiddleName(name));
# //   if (hasFuzzyMiddleNameMatch) {
# //     return hasFuzzyMiddleNameMatch;
# //   }

# //   // Find a match with the middle name missing on the provided aliases
# //   let aliasWithoutMiddleName = knownAliases.map((alias) => nameWithoutMiddleName(alias))
# //   let hasFuzzyAliasMiddleNameMatch = aliasWithoutMiddleName.includes(name);
# //   if (hasFuzzyAliasMiddleNameMatch) {
# //     return hasFuzzyAliasMiddleNameMatch;
# //   }


# //   let hasMiddleInitialMatch = knownAliases.includes(nameWithMiddleInitial(name));
# //    if (hasMiddleInitialMatch) {
# //     return hasMiddleInitialMatch;
# //   }


# //   let aliaswithMiddleInitialName = knownAliases.map((alias) => nameWithMiddleInitial(alias))
# //   let hasAliasMiddleInitialNameMatch = aliaswithMiddleInitialName.includes(name);

# //   if (hasAliasMiddleInitialNameMatch) {
# //     return hasAliasMiddleInitialNameMatch;
# //   }
    
  
# //   return false;
# // }

# // /** Tests **/

# // function assertEqual(expected, result, errorMessage) {
# //   if (result !== expected) {
# //     console.log(errorMessage);
# //     console.log(`expected: ${expected}`);
# //     console.log(`actual: ${result}`);
# //     console.log('');
# //   }
# // }

# // function test() {
# //   let knownAliases;

# //   knownAliases = ["Alphonse Gabriel Capone", "Al Capone"];
# //   assertEqual(true,  nameMatch(knownAliases, "Alphonse Gabriel Capone"), "error 1.1");
# //   assertEqual(true,  nameMatch(knownAliases, "Al Capone"),               "error 1.2");
# //   assertEqual(false, nameMatch(knownAliases, "Alphonse Francis Capone"), "error 1.3");

# //   knownAliases = ["Alphonse Capone"];
# //   assertEqual(true,  nameMatch(knownAliases, "Alphonse Gabriel Capone"), "error 2.1");
# //   assertEqual(true,  nameMatch(knownAliases, "Alphonse Francis Capone"), "error 2.2");
# //   assertEqual(false, nameMatch(knownAliases, "Alexander Capone"),        "error 2.3");

# //   knownAliases = ["Alphonse Gabriel Capone"];
# //   assertEqual(true,  nameMatch(knownAliases, "Alphonse Capone"),         "error 3.1");
# //   assertEqual(false, nameMatch(knownAliases, "Alphonse Francis Capone"), "error 3.2");
# //   assertEqual(false, nameMatch(knownAliases, "Alphonse Edward Capone"),  "error 3.3");

# //   knownAliases = ["Alphonse Gabriel Capone", "Alphonse Francis Capone"];
# //   assertEqual(true,  nameMatch(knownAliases, "Alphonse Gabriel Capone"), "error 4.1");
# //   assertEqual(true,  nameMatch(knownAliases, "Alphonse Francis Capone"), "error 4.2");
# //   assertEqual(false, nameMatch(knownAliases, "Alphonse Edward Capone"),  "error 4.3");

# //   knownAliases = ["Alphonse Gabriel Capone", "Alphonse F Capone"];
# //   assertEqual(true,  nameMatch(knownAliases, "Alphonse G Capone"),       "error 5.1");
# //   assertEqual(true,  nameMatch(knownAliases, "Alphonse Francis Capone"), "error 5.2");
# //   assertEqual(false, nameMatch(knownAliases, "Alphonse E Capone"),       "error 5.3");
# //   assertEqual(false, nameMatch(knownAliases, "Alphonse Edward Capone"),  "error 5.4");
# //   assertEqual(false, nameMatch(knownAliases, "Alphonse Gregory Capone"), "error 5.5");

# //   knownAliases = ["Alphonse Gabriel Capone"];
# //   assertEqual(true,  nameMatch(knownAliases, "Gabriel Alphonse Capone"), "error 6.1");
# //   assertEqual(true,  nameMatch(knownAliases, "Gabriel A Capone"),        "error 6.2");
# //   assertEqual(true,  nameMatch(knownAliases, "Gabriel Capone"),          "error 6.3");
# //   assertEqual(false, nameMatch(knownAliases, "Gabriel Francis Capone"),  "error 6.4");

# //   knownAliases = ["Alphonse Gabriel Capone"];
# //   assertEqual(false, nameMatch(knownAliases, "Alphonse Capone Gabriel"), "error 7.1");
# //   assertEqual(false, nameMatch(knownAliases, "Capone Alphonse Gabriel"), "error 7.2");
# //   assertEqual(false, nameMatch(knownAliases, "Capone Gabriel"),          "error 7.3");

# //   console.log('Test run finished')
# // }
# // test();
def nameMatch(knownAliases, name):
    rec_parts = name.split(" ")
    if len(rec_parts) < 2:
        return False
    
    for name_str in knownAliases:
        if name_str == name:
            return True 

        alia_parts = name_str.split(" ")
    
        if alia_parts[-1] != rec_parts[-1]:
            return False
        
        if compare(alia_parts[:-1], rec_parts[:-1]):
            return True

    return False 

def compare(alias, rec):
    
    n, m = len(alias), len(rec)
    if n == 2 and m < 2:     
        return alias[0] == rec[0] or alias[1] == rec[0]
    if m == 2 and n < 2:
        return alias[0] == rec[0] or rec[1] == alias[0]
    if m < 2 and n < 2:
        return alias[0] == rec[0]
    
    alias_fname, alias_mname = alias[0], alias[1]
    rec_fname, rec_mname = rec[0], rec[1]
    
    #both alias and record is full name  
    if alias == rec or (alias_fname == rec_mname and alias_mname == rec_fname) or (rec_fname == alias_mname and rec_mname == alias_fname): 
        return True 
    
    # when middle name is a initial []
    if len(alias_mname) == 1:
        return fullNameCompare(alias_fname, alias_mname, rec_fname, rec_mname)
    
    if len(alias_mname) == 1:
        return fullNameCompare(rec_fname, rec_mname, alias_fname, alias_mname)
        return (alias_fname == rec_fname and alias_mname == rec_mname[0])or\
        (alias_fname == rec_mname and alias_mname == rec_fname[0])
        
    return False 

def fullNameCompare(firstFname, middleWithInitial, secondFname, middleFull):
    return (firstFname == secondFname and middleWithInitial == middleFull[0])or\
        (firstFname == secondFname and middleWithInitial == middleFull[0])