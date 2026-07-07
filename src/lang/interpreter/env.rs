use super::value::Value;
use crate::lang::error::{HayashiError, Result};
use std::collections::{HashMap, HashSet};

struct Scope {
    vars: HashMap<String, Value>,
    consts: HashSet<String>,
}

impl Scope {
    fn new() -> Self {
        Self {
            vars: HashMap::new(),
            consts: HashSet::new(),
        }
    }
}

pub struct Env {
    scopes: Vec<Scope>,
    pub(crate) quiet_mode: bool,
    quiet_stack: Vec<bool>,
}

impl Env {
    pub fn new() -> Self {
        Self {
            scopes: vec![Scope::new()],
            quiet_mode: false,
            quiet_stack: Vec::new(),
        }
    }

    pub fn push_scope(&mut self) {
        self.quiet_stack.push(self.quiet_mode);
        self.scopes.push(Scope::new());
    }

    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
            self.quiet_mode = self.quiet_stack.pop().unwrap_or(false);
        }
    }

    pub fn quiet_mode(&self) -> bool {
        self.quiet_mode
    }

    pub fn set_quiet_mode(&mut self, mode: bool) {
        self.quiet_mode = mode;
    }

    pub fn declare(&mut self, name: &str, val: Value) -> Result<()> {
        for scope in self.scopes.iter().rev() {
            if scope.consts.contains(name) {
                return Err(HayashiError::Runtime(format!(
                    "cannot redeclare const '{name}'"
                )));
            }
        }
        self.scopes
            .last_mut()
            .unwrap()
            .vars
            .insert(name.to_string(), val);
        Ok(())
    }

    pub fn declare_const(&mut self, name: &str, val: Value) {
        let scope = self.scopes.last_mut().unwrap();
        scope.vars.insert(name.to_string(), val);
        scope.consts.insert(name.to_string());
    }

    pub fn set(&mut self, name: &str, val: Value) -> Result<()> {
        for scope in self.scopes.iter().rev() {
            if scope.consts.contains(name) {
                return Err(HayashiError::Runtime(format!(
                    "cannot reassign const '{name}'"
                )));
            }
        }
        for scope in self.scopes.iter_mut().rev() {
            if scope.vars.contains_key(name) {
                scope.vars.insert(name.to_string(), val);
                return Ok(());
            }
        }
        self.scopes
            .last_mut()
            .unwrap()
            .vars
            .insert(name.to_string(), val);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.vars.get(name) {
                return Some(v);
            }
        }
        None
    }

    pub fn all_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .scopes
            .iter()
            .flat_map(|s| s.vars.keys().cloned())
            .collect();
        names.sort();
        names.dedup();
        names
    }

    pub fn remove(&mut self, name: &str) {
        for scope in self.scopes.iter_mut().rev() {
            if scope.vars.remove(name).is_some() {
                scope.consts.remove(name);
                return;
            }
        }
    }

    pub fn var_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for scope in self.scopes.iter().rev() {
            for key in scope.vars.keys() {
                if !names.contains(key) {
                    names.push(key.clone());
                }
            }
        }
        names
    }
}
